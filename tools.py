"""Definición de tools (schema OpenAI/Groq) y handlers.

- Sin SUPABASE_*: stubs locales.
- Con Supabase: tablas/columnas vía variables ``ERP_SUPABASE_*`` (un deploy = un proyecto DB).

Ejemplo mínimo de tablas demo: ``erp_sales`` (fecha, monto), ``erp_products`` (id, nombre, precio).

Ventas: ``ERP_SUPABASE_SALES_SOURCE=orders`` o ``items``; listado de OV: ``list_sales_orders``.

Compras: ``get_purchase_summary``, ``list_purchase_orders``, ``count_purchase_orders_by_status``, ``list_purchase_order_items``.

Proveedores: ``search_suppliers`` + ``ERP_SUPABASE_SUPPLIERS_*`` / ``ERP_SUPABASE_SUPPLIER_COL_*``.

El LLM no recibe credenciales de Supabase.
"""

from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from datetime import date, datetime, timedelta
from typing import Any

# Schemas que entiende Groq (mismo formato que OpenAI tool calling)
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": (
                "Resumen numérico agregado de órdenes de venta (total $, cantidad de órdenes) para un rango YYYY-MM-DD. "
                "SOLO para consultas específicas de órdenes de venta. "
                "NO devuelve clientes, NO devuelve órdenes individuales, NO devuelve detalle. "
                "PREFERIR get_invoice_summary cuando el usuario pregunte por 'ventas', 'total vendido', 'cuánto vendimos'. "
                "Usar este solo si el usuario pide explícitamente resumen de 'órdenes de venta' o 'OV'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive, formato YYYY-MM-DD",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive, formato YYYY-MM-DD",
                    },
                },
                "required": ["desde", "hasta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sales_orders",
            "description": (
                "Lista ÓRDENES DE VENTA (pedidos, OV) con cliente, monto y estado para un rango de fechas. "
                "Usar SOLO cuando el usuario diga explícitamente: 'órdenes de venta', 'OV', 'pedidos del día', 'pedidos'. "
                "NO usar si el usuario dice simplemente 'ventas' o 'qué se vendió': para eso usar list_customer_invoices. "
                "Acepta filtro opcional por cliente (customer_name). "
                "Si pide 'todas las órdenes' o 'todo el detalle' sin filtro temporal, NO llamar: pedirle al usuario que acote el período."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; si falta se infiere)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Nombre o razón social del cliente para filtrar (opcional)",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "UUID del cliente para filtrar (opcional; alternativa a customer_name)",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de órdenes a devolver (default 30, máx 200)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_sales_orders_by_status",
            "description": (
                "Cuenta órdenes de venta agrupadas por estado en un rango de fechas. "
                "Si el usuario no da fechas, llamar igual: se usa un rango por defecto (últimos días). "
                "Usar cuando pidan contabilizar/cuantas órdenes hay por estado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; si falta se infiere)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_selling_products",
            "description": (
                "Obtiene el ranking de productos más vendidos en un rango de fechas (por cantidad y monto). "
                "Incluye estimación de costo de mercadería (basado en cost_price del producto), utilidad estimada y markup por producto. "
                "Usar cuando el usuario pregunte: 'productos más vendidos', 'top de ventas', 'qué se vendió más', "
                "'qué producto me deja más ganancia' (mostrar ranking con margen estimado y aclarar que para ranking exacto por ganancia debe ir al Reporte de Rentabilidad del ERP), "
                "'cuánto gano con X producto'. "
                "IMPORTANTE: el margen es una ESTIMACIÓN basada en el cost_price actual del producto; "
                "para margen exacto del período usar get_profit_margin_summary o el Reporte de Rentabilidad del ERP."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; si falta se infiere)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de productos en el ranking (default 10, máx 100)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_least_selling_products",
            "description": (
                "Obtiene ranking de productos menos vendidos en un rango de fechas a partir de líneas de venta. "
                "Devuelve productos con menor cantidad y monto vendido."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; si falta se infiere)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de productos en el ranking (default 10, máx 100)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_purchase_summary",
            "description": (
                "Resumen agregado de compras (suma totales de órdenes de compra) entre dos fechas YYYY-MM-DD. "
                "Usar cuando pregunten cuánto compramos, total de OCs en un período."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD",
                    },
                },
                "required": ["desde", "hasta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_purchase_orders",
            "description": (
                "Lista órdenes de compra (OC): número, fechas, total, estado, etc. "
                "SOLO llamar si el usuario especificó un período concreto. Si pide 'todas las compras' sin filtro, NO llamar: pedirle que acote el período. "
                "Usar para listados de compras, OCs pendientes, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de filas (default 30, máx 200)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_purchase_orders_by_status",
            "description": (
                "Cuenta órdenes de compra por estado en un rango de fechas. "
                "Si no dan fechas, rango por defecto. "
                "Usar cuando pidan contabilizar OCs por estado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_purchase_order_items",
            "description": (
                "Lista ítems/líneas de una orden de compra por su id (UUID). "
                "Usar cuando pregunten qué trae una OC, detalle de líneas, productos de la compra."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "purchase_order_id": {
                        "type": "string",
                        "description": "UUID de la orden de compra (purchase_orders.id)",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de líneas (default 100, máx 500)",
                    },
                },
                "required": ["purchase_order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_available_stock",
            "description": (
                "Consulta stock de un producto por nombre/código/SKU/barcode. "
                "Devuelve: stock total, stock reservado, stock disponible (total-reservado), "
                "stock proyectado (columna stock_projected de warehouse_stock) y stock mínimo. "
                "Usar cuando pregunten por stock disponible, stock proyectado, stock actual, "
                "cuánto hay de un artículo, o cuánto queda de un producto."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre, código, SKU, barcode o texto del producto",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de coincidencias de producto (default 5, máx 20)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_products_below_min_stock",
            "description": (
                "Lista productos con stock disponible por debajo (o igual) del mínimo/punto de pedido "
                "comparando warehouse_stock contra min_stock."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de filas (default 50, máx 200)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_top_products_by_stock",
            "description": (
                "Lista los N productos con mayor (o menor) cantidad de stock disponible. "
                "Usar cuando pregunten cuál es el producto con más stock, top de inventario, "
                "qué artículo tiene mayor/menor cantidad disponible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Cantidad de productos a devolver (default 10, máx 100)",
                    },
                    "order": {
                        "type": "string",
                        "description": "'desc' para mayor stock primero (default), 'asc' para menor stock primero",
                    },
                    "warehouse_name": {
                        "type": "string",
                        "description": "Nombre del almacén a consultar (opcional; si se omite suma todos los depósitos)",
                    },
                    "only_principal": {
                        "type": "boolean",
                        "description": "Si true, filtra solo el depósito de tipo 'principal' (default false)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_recent_product_movements",
            "description": (
                "Muestra ingresos/egresos recientes de un artículo usando cardex_report. "
                "Por defecto trae la última semana."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre, SKU/código, barcode o texto del artículo",
                    },
                    "days": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Días hacia atrás (default 7, máx 90)",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de movimientos por consulta (default 100, máx 300)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Busca productos por texto (SKU/código, nombre, etc.). "
                "Requiere un texto de búsqueda concreto; NO llamar si el usuario pide 'todos los productos' sin búsqueda. "
                "Devuelve resultados con sku_o_codigo, nombre, precio (y id UUID si aplica). "
                "Usar cuando pregunten por productos, nombre por SKU, listados filtrados."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto de búsqueda",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de resultados (default 10). Puede ser número o string.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_suppliers",
            "description": (
                "Busca proveedores por texto (razón social, CUIT/tax_id, nombre de contacto, etc.). "
                "Requiere un texto de búsqueda; NO llamar si el usuario pide 'todos los proveedores' sin filtro. "
                "Devuelve id, razon_social, tax_id, contacto, email, telefono, activo. "
                "Usar cuando pregunten por proveedores, CUIT, quién vende, datos de un supplier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto de búsqueda",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de resultados (default 10). Puede ser número o string.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_customers",
            "description": (
                "Busca clientes por texto (razón social, CUIT/tax_id, nombre de contacto, código interno, etc.). "
                "Requiere un texto de búsqueda; NO llamar si el usuario pide 'todos los clientes' sin filtro. "
                "Devuelve id, razon_social, tax_id, contacto, codigo, email, telefono, activo. "
                "Usar cuando pregunten por clientes, CUIT de cliente, datos de un customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto de búsqueda",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de resultados (default 10). Puede ser número o string.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_balance",
            "description": (
                "Devuelve el saldo de cuenta corriente de un cliente: cuánto debe (total_debt > 0) "
                "o si tiene saldo a favor (total_debt < 0). "
                "Usa la vista 'customer_balances_view' de Supabase, igual que el ERP. "
                "Usar cuando pregunten: 'saldo de cuenta corriente', 'cuánto debe', 'saldo del cliente', "
                "'deuda del cliente', 'tiene saldo a favor', 'estado de cuenta', 'cuánto le debemos'. "
                "El campo total_debt positivo indica deuda; negativo indica crédito a favor del cliente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "string",
                        "description": "Nombre, CUIT o UUID del cliente a consultar.",
                    },
                },
                "required": ["customer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sales_order_items",
            "description": (
                "Lista los ítems/líneas de una orden de venta específica. "
                "Usar cuando pregunten qué productos compró un cliente en una venta, "
                "detalle de líneas, productos de una OV. "
                "Requiere el id (UUID) de la orden de venta obtenido de list_sales_orders, "
                "o el número de orden (ej: OV-2026-003400)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sales_order_id": {
                        "type": "string",
                        "description": "UUID de la orden de venta (sales_orders.id)",
                    },
                    "order_number": {
                        "type": "string",
                        "description": "Número de orden de venta (ej: OV-2026-003400). Alternativa al UUID.",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Máximo de líneas (default 100, máx 500)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customer_invoices",
            "description": (
                "Lista facturas/ventas de clientes (número, fecha, total, cobrado, saldo, estado) en un rango de fechas. "
                "Acepta filtro opcional por cliente (nombre o UUID) y/o estado. "
                "Usar SOLO cuando el usuario pida EXPLÍCITAMENTE un listado o detalle de facturas individuales: "
                "'mostrá las ventas', 'listado de facturas', 'qué facturas hay', 'detalle de ventas', "
                "'facturas pendientes/vencidas', 'ventas de un cliente específico', 'comprobantes emitidos'. "
                "NO usar si el usuario solo pregunta cuánto se vendió, el total, o el monto del día/semana/mes: "
                "para eso usar get_invoice_summary. "
                "LÍMITE: pasar siempre limit=10 salvo que el usuario pida explícitamente más registros. "
                "IMPORTANTE: 'ventas' para el usuario = facturación real (customer_invoices), NO órdenes de venta. "
                "Excluye automáticamente: cancelled, voided, converted, draft y tipos nota_pedido/np."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (issue_date)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (default hoy)",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Nombre o razón social del cliente (opcional)",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "UUID del cliente (opcional; alternativa a customer_name)",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filtrar por estado: pending, paid, partial, overdue, cancelled, voided (opcional)",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de facturas a devolver. SIEMPRE usar 10 salvo que el usuario pida explícitamente más. Máx absoluto 200.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_invoice_summary",
            "description": (
                "Resumen agregado de facturación/ventas: total facturado, total cobrado, saldo pendiente y cantidad de facturas. "
                "Acepta filtro opcional por cliente. "
                "Es la función PRINCIPAL para cualquier consulta de ventas sin pedir listado explícito. "
                "Usar cuando pregunten: 'ventas de hoy', 'ventas del día', 'cuánto se vendió hoy', "
                "'cuántas ventas hubo hoy', 'ventas de ayer', 'ventas de esta semana', 'ventas del mes', "
                "'cuánto facturamos', 'total de ventas', 'monto total de ventas', "
                "'cuánto vendimos', 'total vendido', 'cuánto cobró un cliente', 'saldo total de deuda', "
                "'resumen de ventas', 'resumen de facturación'. "
                "IMPORTANTE: 'ventas' para el usuario = facturación real (customer_invoices). "
                "Excluye automáticamente: cancelled, voided, converted, draft, nota_pedido/np y facturas consolidadas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Nombre o razón social del cliente (opcional)",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "UUID del cliente (opcional)",
                    },
                },
                "required": ["desde", "hasta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_profit_margin_summary",
            "description": (
                "Calcula el margen de ganancia real del período: costo de mercadería (con IVA), "
                "utilidad, markup (ganancia / costo × 100) y porcentaje de utilidad sobre ventas. "
                "El costo se obtiene de los ítems de cada factura (purchase_cost por línea, "
                "con fallback al cost_price del producto), igual que muestra el reporte del ERP. "
                "Usar cuando el usuario pregunte: 'margen de ganancia', 'cuánto gané realmente', "
                "'cuál es mi rentabilidad', 'ganancia real', 'utilidad del período', "
                "'costo de mercadería del mes', 'markup del período', 'cuánto gané', "
                "'cuánto es la ganancia', 'margen real', 'rentabilidad real'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Nombre o razón social del cliente (opcional)",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "UUID del cliente (opcional)",
                    },
                },
                "required": ["desde", "hasta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customer_invoice_items",
            "description": (
                "Lista los ítems/líneas de una factura de cliente específica. "
                "Usar cuando pregunten qué productos tiene una factura, detalle de una FC, líneas de comprobante."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "UUID de la factura (customer_invoices.id)",
                    },
                    "invoice_number": {
                        "type": "string",
                        "description": "Número de factura (ej: FC-0001-00000123). Alternativa al UUID.",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de líneas (default 100, máx 500)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customer_payments",
            "description": (
                "Lista pagos registrados para una factura o cliente. "
                "Usar cuando pregunten por cobros, pagos recibidos, método de pago, historial de pagos de una factura."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "UUID de la factura para filtrar pagos (opcional)",
                    },
                    "invoice_number": {
                        "type": "string",
                        "description": "Número de factura para filtrar pagos (opcional)",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Nombre del cliente para ver todos sus pagos (opcional)",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "UUID del cliente (opcional)",
                    },
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (payment_date)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de pagos (default 50, máx 200)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_products_without_stock_days",
            "description": (
                "Lista productos que tienen stock = 0 y cuyo registro no fue actualizado en los últimos N días. "
                "Usar cuando pregunten: 'productos sin stock los últimos X días', 'productos que no tuvieron stock', "
                "'artículos con stock cero hace más de X días', 'qué productos llevan X días sin stock'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Cantidad de días sin stock (default 15)",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de filas (default 50, máx 200)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_sellers",
            "description": (
                "Ranking de vendedores basado en ÓRDENES DE VENTA (sales_orders) — NO en facturas. "
                "Incluye ventas pendientes, en proceso y entregadas. "
                "Usar SOLO cuando pregunten por: 'órdenes de venta por vendedor', 'quien hizo más órdenes', "
                "'ranking de ventas' (en términos de pedidos/órdenes), 'mejor vendedor por órdenes', 'ventas de un vendedor'. "
                "Si el usuario nombra un vendedor específico, usar limit alto (hasta 50) para poder ubicarlo en el ranking. "
                "NO usar si el usuario pide 'facturación real', 'quién más facturó' o 'facturas emitidas': "
                "para eso usar get_top_sellers_by_invoicing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; se infiere si falta)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Métrica de ranking: 'monto' (suma total_amount, default) o 'cantidad' (conteo de órdenes)",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de vendedores a devolver (default 10, máx 50)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_sellers_by_invoicing",
            "description": (
                "Ranking de VENDEDORES (no clientes) basado en FACTURACIÓN REAL (customer_invoices: FA, FB, remito). "
                "Solo incluye comprobantes ya emitidos. "
                "Devuelve por vendedor: total_facturado, costo_mercaderia (c/ IVA), utilidad, markup_pct y pct_utilidad_ventas. "
                "Usar cuando pregunten: 'quién más facturó', 'vendedor con mayor facturación', "
                "'qué vendedor genera más rentabilidad', 'qué vendedor deja más ganancia', "
                "'ranking de vendedores por ganancia', 'facturación real por vendedor'. "
                "Si el usuario nombra un vendedor específico, usar limit alto (hasta 50) para poder ubicarlo en el ranking. "
                "NO usar para órdenes de venta pendientes: para eso usar get_top_sellers. "
                "NO usar si el usuario pregunta por CLIENTES: para eso usar get_top_customers_by_invoicing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; se infiere si falta)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Métrica: 'monto' (suma total_amount, default) o 'cantidad' (conteo de facturas)",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de vendedores a devolver (default 10, máx 50)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_customers_by_debt",
            "description": (
                "Ranking de clientes ordenados por SALDO DE DEUDA (cuenta corriente), de mayor a menor. "
                "Usa la vista customer_balances_view, igual que el ERP. "
                "Usar cuando pregunten: 'qué cliente debe más', 'cliente con mayor deuda', "
                "'ranking de deudores', 'quién tiene mayor saldo pendiente', 'top deudores', "
                "'cuáles son los clientes más endeudados', 'quién me debe más'. "
                "NO usar para ranking de facturación: para eso usar get_top_customers_by_invoicing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de clientes a devolver (default 10, máx 50)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_customers_by_invoicing",
            "description": (
                "Ranking de CLIENTES por facturación acumulada real (customer_invoices). "
                "Suma el total facturado por cliente en el período, excluyendo canceladas, anuladas, borradores y facturas consolidadas. "
                "Usar cuando pregunten: 'cliente con mayor facturación', 'qué cliente compró más', "
                "'ranking de clientes', 'mejor cliente', 'clientes con más compras', 'top clientes'. "
                "NO confundir con vendedores: si el usuario dice 'vendedor' usar get_top_sellers_by_invoicing. "
                "NO usar para ranking de deuda: para eso usar get_top_customers_by_debt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; se infiere si falta)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Métrica: 'monto' (suma total_amount, default) o 'cantidad' (conteo de facturas)",
                    },
                    "limit": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Máximo de clientes a devolver (default 10, máx 50)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_sales_units",
            "description": (
                "Consulta cuántas unidades/kg se vendieron (facturaron) de un producto específico en un rango de fechas. "
                "Acepta nombre parcial, SKU, código o cualquier texto que identifique el producto (búsqueda no exacta). "
                "Devuelve por producto: nombre, SKU/código, sale_unit, cantidad_vendida, kg_vendidos, "
                "monto_vendido, costo_mercaderia (c/ IVA, exacto desde purchase_cost del ítem o cost_price del producto), "
                "utilidad, markup_pct y pct_utilidad_ventas, y cantidad_ventas. "
                "Usar cuando el usuario pregunte: '¿cuánto se vendió de X?', '¿cuántas unidades de X?', "
                "'¿cuántos kilos de X?', 'ventas del producto X', 'cuánto facturamos de X', "
                "'cuánto gano con X', 'margen de X', 'ganancia de X', 'rentabilidad de X', "
                "'qué cantidad se vendió de X', 'total de X vendido'. "
                "Si el usuario no especifica fechas, usar los últimos 30 días. "
                "Si la búsqueda devuelve varios productos similares, se muestran todos con sus cantidades y márgenes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre, SKU, código o texto que identifique el producto (búsqueda parcial/no exacta)",
                    },
                    "desde": {
                        "type": "string",
                        "description": "Fecha inicio inclusive YYYY-MM-DD (opcional; si no se provee, se usan los últimos 30 días)",
                    },
                    "hasta": {
                        "type": "string",
                        "description": "Fecha fin inclusive YYYY-MM-DD (opcional; default hoy)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ContextVar para override por request (multi-tenant, async-safe)
_request_supabase_override: ContextVar[Any | None] = ContextVar("_request_supabase_override", default=None)

# Caché de clientes por db_key
_supabase_client_by_key: dict[str, Any] = {}


def data_backend_label() -> str:
    return "supabase" if _request_supabase_override.get() is not None else "stub"


def _get_supabase():
    """Solo devuelve el cliente si se seteó un override vía set_request_supabase()."""
    return _request_supabase_override.get()


def _get_supabase_for_key(db_key: str) -> Any | None:
    """Devuelve cliente Supabase para el tenant db_key, o None si no hay credenciales.
    
    Busca SUPABASE_URL_{DB_KEY} y SUPABASE_SERVICE_ROLE_KEY_{DB_KEY} en el entorno.
    Ejemplo: db_key='ALPINA_PROD' → SUPABASE_URL_ALPINA_PROD + SUPABASE_SERVICE_ROLE_KEY_ALPINA_PROD
    """
    safe = db_key.strip().upper()
    if safe in _supabase_client_by_key:
        return _supabase_client_by_key[safe]
    url = os.environ.get(f"SUPABASE_URL_{safe}", "").strip()
    key = os.environ.get(f"SUPABASE_SERVICE_ROLE_KEY_{safe}", "").strip()
    if not url or not key:
        return None
    from supabase import create_client
    client = create_client(url, key)
    _supabase_client_by_key[safe] = client
    return client


def set_request_supabase(client: Any | None):
    """Establece el cliente Supabase activo para el request actual (ContextVar).
    
    Retorna el Token para poder resetear con _request_supabase_override.reset(token).
    """
    return _request_supabase_override.set(client)


def _sales_table() -> str:
    return os.environ.get("ERP_SUPABASE_SALES_TABLE", "erp_sales").strip() or "erp_sales"


def _sales_orders_column_config() -> tuple[str, str, str]:
    """(tabla, col_fecha, col_importe)."""
    table = _sales_table()
    date_c = (os.environ.get("ERP_SUPABASE_SALES_DATE_COL", "fecha") or "fecha").strip()
    amount_c = (os.environ.get("ERP_SUPABASE_SALES_AMOUNT_COL", "monto") or "monto").strip()
    if not _safe_sql_identifier(table):
        raise ValueError(f"ERP_SUPABASE_SALES_TABLE inválido: {table!r}")
    for c in (date_c, amount_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"columna de ventas inválida: {c!r}")
    return table, date_c, amount_c


def _sales_source_mode() -> str:
    v = os.environ.get("ERP_SUPABASE_SALES_SOURCE", "orders").strip().lower()
    if v in ("items", "lines", "order_lines", "sales_order_items"):
        return "items"
    return "orders"


def _sales_items_column_config() -> tuple[str, str, str, str, str, str]:
    """(items_table, amount_col, fk_col, orders_table, order_date_col, order_id_col)."""
    items_t = (
        os.environ.get("ERP_SUPABASE_SALES_ITEMS_TABLE", "sales_order_items") or "sales_order_items"
    ).strip()
    amount_c = (
        os.environ.get("ERP_SUPABASE_SALES_ITEMS_AMOUNT_COL", "line_total") or "line_total"
    ).strip()
    fk_c = (os.environ.get("ERP_SUPABASE_SALES_ITEMS_FK_COL", "sales_order_id") or "sales_order_id").strip()
    orders_t = (os.environ.get("ERP_SUPABASE_SALES_ORDERS_TABLE", "sales_orders") or "sales_orders").strip()
    date_c = (os.environ.get("ERP_SUPABASE_SALES_ORDER_DATE_COL", "order_date") or "order_date").strip()
    oid_c = (os.environ.get("ERP_SUPABASE_SALES_ORDER_ID_COL", "id") or "id").strip()
    for name in (items_t, amount_c, fk_c, orders_t, date_c, oid_c):
        if not _safe_sql_identifier(name):
            raise ValueError(f"identificador ventas (modo items) inválido: {name!r}")
    return items_t, amount_c, fk_c, orders_t, date_c, oid_c


def _sales_items_top_columns() -> tuple[str, str]:
    product_c = (os.environ.get("ERP_SUPABASE_SALES_ITEMS_PRODUCT_ID_COL", "product_id") or "product_id").strip()
    qty_c = (os.environ.get("ERP_SUPABASE_SALES_ITEMS_QTY_COL", "quantity") or "quantity").strip()
    for c in (product_c, qty_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"columna de ítems de venta inválida para top productos: {c!r}")
    return product_c, qty_c


def _products_table() -> str:
    return os.environ.get("ERP_SUPABASE_PRODUCTS_TABLE", "erp_products").strip() or "erp_products"


def _suppliers_table() -> str:
    return os.environ.get("ERP_SUPABASE_SUPPLIERS_TABLE", "suppliers").strip() or "suppliers"


def _customers_table() -> str:
    return os.environ.get("ERP_SUPABASE_CUSTOMERS_TABLE", "customers").strip() or "customers"


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _supplier_column_config() -> tuple[str, str, str, str, str, str, str, str | None]:
    """tabla, id, razon_social, tax_id, email, phone, is_active, contacto (opcional)."""
    table = _suppliers_table()
    id_c = (os.environ.get("ERP_SUPABASE_SUPPLIER_COL_ID", "id") or "id").strip()
    name_c = (
        os.environ.get("ERP_SUPABASE_SUPPLIER_COL_BUSINESS_NAME", "business_name") or "business_name"
    ).strip()
    tax_c = (os.environ.get("ERP_SUPABASE_SUPPLIER_COL_TAX_ID", "tax_id") or "tax_id").strip()
    email_c = (os.environ.get("ERP_SUPABASE_SUPPLIER_COL_EMAIL", "email") or "email").strip()
    phone_c = (os.environ.get("ERP_SUPABASE_SUPPLIER_COL_PHONE", "phone") or "phone").strip()
    active_c = (os.environ.get("ERP_SUPABASE_SUPPLIER_COL_ACTIVE", "is_active") or "is_active").strip()
    contact_raw = (os.environ.get("ERP_SUPABASE_SUPPLIER_COL_CONTACT", "contact_name") or "").strip()
    contact_c: str | None = contact_raw if contact_raw else None
    if not _safe_sql_identifier(table):
        raise ValueError(f"ERP_SUPABASE_SUPPLIERS_TABLE inválido: {table!r}")
    for c in (id_c, name_c, tax_c, email_c, phone_c, active_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"columna de proveedor inválida: {c!r}")
    if contact_c is not None and not _safe_sql_identifier(contact_c):
        raise ValueError(f"columna contacto inválida: {contact_c!r}")
    return table, id_c, name_c, tax_c, email_c, phone_c, active_c, contact_c


def _customer_column_config() -> tuple[str, str, str, str, str, str, str, str, str | None]:
    """tabla, id, razon_social, tax_id, email, phone, is_active, codigo, contacto (opcional)."""
    table = _customers_table()
    id_c = (os.environ.get("ERP_SUPABASE_CUSTOMER_COL_ID", "id") or "id").strip()
    name_c = (
        os.environ.get("ERP_SUPABASE_CUSTOMER_COL_BUSINESS_NAME", "business_name") or "business_name"
    ).strip()
    tax_c = (os.environ.get("ERP_SUPABASE_CUSTOMER_COL_TAX_ID", "tax_id") or "tax_id").strip()
    email_c = (os.environ.get("ERP_SUPABASE_CUSTOMER_COL_EMAIL", "email") or "email").strip()
    phone_c = (os.environ.get("ERP_SUPABASE_CUSTOMER_COL_PHONE", "phone") or "phone").strip()
    active_c = (os.environ.get("ERP_SUPABASE_CUSTOMER_COL_ACTIVE", "is_active") or "is_active").strip()
    code_c = (
        os.environ.get("ERP_SUPABASE_CUSTOMER_COL_INTERNAL_CODE", "internal_code") or "internal_code"
    ).strip()
    contact_raw = (os.environ.get("ERP_SUPABASE_CUSTOMER_COL_CONTACT", "contact_name") or "").strip()
    contact_c: str | None = contact_raw if contact_raw else None
    if not _safe_sql_identifier(table):
        raise ValueError(f"ERP_SUPABASE_CUSTOMERS_TABLE inválido: {table!r}")
    for c in (id_c, name_c, tax_c, email_c, phone_c, active_c, code_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"columna de cliente inválida: {c!r}")
    if contact_c is not None and not _safe_sql_identifier(contact_c):
        raise ValueError(f"columna contacto cliente inválida: {contact_c!r}")
    return table, id_c, name_c, tax_c, email_c, phone_c, active_c, code_c, contact_c


def _supplier_select_columns(
    id_c: str,
    name_c: str,
    tax_c: str,
    email_c: str,
    phone_c: str,
    active_c: str,
    contact_c: str | None,
    extra_cols: list[str],
) -> str:
    parts: list[str] = []
    for c in (id_c, name_c, tax_c, contact_c, email_c, phone_c, active_c):
        if c and c not in parts:
            parts.append(c)
    for col in extra_cols:
        if col not in parts:
            parts.append(col)
    return ",".join(parts)


def _customer_select_columns(
    id_c: str,
    name_c: str,
    tax_c: str,
    email_c: str,
    phone_c: str,
    active_c: str,
    code_c: str,
    contact_c: str | None,
    extra_cols: list[str],
) -> str:
    parts: list[str] = []
    for c in (id_c, name_c, tax_c, code_c, contact_c, email_c, phone_c, active_c):
        if c and c not in parts:
            parts.append(c)
    for col in extra_cols:
        if col not in parts:
            parts.append(col)
    return ",".join(parts)


def _supplier_extra_col_names() -> list[str]:
    extra = (os.environ.get("ERP_SUPABASE_SUPPLIER_EXTRA_COLS", "") or "").strip()
    if not extra:
        return []
    out: list[str] = []
    for raw in extra.split(","):
        col = raw.strip()
        if not col or col in out:
            continue
        if not _safe_sql_identifier(col):
            raise ValueError(f"ERP_SUPABASE_SUPPLIER_EXTRA_COLS: columna inválida {col!r}")
        out.append(col)
    return out


def _customer_extra_col_names() -> list[str]:
    extra = (os.environ.get("ERP_SUPABASE_CUSTOMER_EXTRA_COLS", "") or "").strip()
    if not extra:
        return []
    out: list[str] = []
    for raw in extra.split(","):
        col = raw.strip()
        if not col or col in out:
            continue
        if not _safe_sql_identifier(col):
            raise ValueError(f"ERP_SUPABASE_CUSTOMER_EXTRA_COLS: columna inválida {col!r}")
        out.append(col)
    return out


def _normalize_supplier_row(
    row: dict[str, Any],
    id_c: str,
    name_c: str,
    tax_c: str,
    email_c: str,
    phone_c: str,
    active_c: str,
    contact_c: str | None,
    extra_cols: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": row.get(id_c),
        "razon_social": row.get(name_c),
        "tax_id": row.get(tax_c),
        "email": row.get(email_c),
        "telefono": row.get(phone_c),
        "activo": row.get(active_c),
    }
    if contact_c:
        out["contacto"] = row.get(contact_c)
    for ec in extra_cols:
        if ec in row:
            out[ec] = row[ec]
    return out


def _normalize_customer_row(
    row: dict[str, Any],
    id_c: str,
    name_c: str,
    tax_c: str,
    email_c: str,
    phone_c: str,
    active_c: str,
    code_c: str,
    contact_c: str | None,
    extra_cols: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": row.get(id_c),
        "razon_social": row.get(name_c),
        "tax_id": row.get(tax_c),
        "codigo": row.get(code_c),
        "email": row.get(email_c),
        "telefono": row.get(phone_c),
        "activo": row.get(active_c),
    }
    if contact_c:
        out["contacto"] = row.get(contact_c)
    for ec in extra_cols:
        if ec in row:
            out[ec] = row[ec]
    return out


def _safe_sql_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _product_table_columns() -> tuple[str, str, str, str | None]:
    """(col_codigo_o_sku, col_nombre, col_precio, col_uuid_opcional)."""
    code = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_CODE", "id") or "id").strip()
    name = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_NAME", "nombre") or "nombre").strip()
    price = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_PRICE", "precio") or "precio").strip()
    uuid_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_UUID", "") or "").strip() or None
    for c in (code, name, price):
        if not _safe_sql_identifier(c):
            raise ValueError(f"nombre de columna inválido en productos: {c!r}")
    if uuid_col is not None and not _safe_sql_identifier(uuid_col):
        raise ValueError(f"nombre de columna inválido (uuid): {uuid_col!r}")
    return code, name, price, uuid_col


def _product_col_sale_unit() -> str:
    c = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_SALE_UNIT", "sale_unit") or "sale_unit").strip()
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PRODUCT_COL_SALE_UNIT inválida: {c!r}")
    return c


def _product_col_unit_weight() -> str:
    c = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_UNIT_WEIGHT", "unit_weight") or "unit_weight").strip()
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PRODUCT_COL_UNIT_WEIGHT inválida: {c!r}")
    return c


def _product_select_list(code: str, name: str, price: str, uuid_col: str | None) -> str:
    parts: list[str] = []
    for c in (uuid_col, code, name, price):
        if c and c not in parts:
            parts.append(c)
    return ",".join(parts)


def _normalize_product_row(
    row: dict[str, Any],
    code_col: str,
    name_col: str,
    price_col: str,
    uuid_col: str | None,
) -> dict[str, Any]:
    raw_price = row.get(price_col)
    try:
        precio = float(raw_price) if raw_price is not None else 0.0
    except (TypeError, ValueError):
        precio = 0.0
    out: dict[str, Any] = {
        "sku_o_codigo": row.get(code_col),
        "nombre": row.get(name_col),
        "precio": precio,
    }
    if uuid_col and uuid_col in row and row.get(uuid_col) is not None:
        out["id"] = row[uuid_col]
    return out


def _to_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None:
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _product_search_columns(
    code_col: str,
    name_col: str,
    uuid_col: str | None,
) -> tuple[str, list[str]]:
    barcode_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_BARCODE", "") or "").strip() or None
    arca_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_ARCA_CODE", "") or "").strip() or None
    for c in (barcode_col, arca_col):
        if c is not None and not _safe_sql_identifier(c):
            raise ValueError(f"columna de búsqueda de producto inválida: {c!r}")

    select_parts: list[str] = []
    for c in (uuid_col, code_col, name_col, barcode_col, arca_col):
        if c and c not in select_parts:
            select_parts.append(c)

    search_cols: list[str] = []
    for c in (name_col, code_col, barcode_col, arca_col):
        if c and c not in search_cols:
            search_cols.append(c)
    return ",".join(select_parts), search_cols


def _resolve_products_by_query(query: str, limit: int) -> list[dict[str, Any]]:
    client = _get_supabase()
    assert client is not None
    table = _products_table()
    code_c, name_c, _, uuid_c = _product_table_columns()
    select_cols, search_cols = _product_search_columns(code_c, name_c, uuid_c)
    print(f"[DEBUG] tabla productos: {table!r}  |  uuid_col: {uuid_c!r}  |  code: {code_c!r}  |  name: {name_c!r}", flush=True)
    if not uuid_c:
        raise ValueError("ERP_SUPABASE_PRODUCT_COL_UUID es obligatorio para consultas de stock/cardex")
    tok = _like_token(query)
    pat = f"%{tok}%"
    or_clause = ",".join(f"{col}.ilike.{pat}" for col in search_cols)
    print(f"[DEBUG] SELECT {select_cols} FROM {table} WHERE {or_clause} LIMIT {limit}", flush=True)
    try:
        r = client.table(table).select(select_cols).or_(or_clause).limit(limit).execute()
        return r.data or []
    except Exception:
        # Fallback robusto si alguna columna opcional no existe o no soporta ilike.
        base_parts: list[str] = []
        for c in (uuid_c, code_c, name_c):
            if c and c not in base_parts:
                base_parts.append(c)
        base_select = ",".join(base_parts)
        try:
            r = (
                client.table(table)
                .select(base_select)
                .or_(f"{name_c}.ilike.{pat},{code_c}.ilike.{pat}")
                .limit(limit)
                .execute()
            )
            return r.data or []
        except Exception:
            name_rows = (
                client.table(table).select(base_select).ilike(name_c, pat).limit(limit).execute().data or []
            )
            if query.strip():
                code_rows = (
                    client.table(table).select(base_select).eq(code_c, query.strip()).limit(limit).execute().data
                    or []
                )
                by_id: dict[Any, dict[str, Any]] = {}
                for row in name_rows + code_rows:
                    rid = row.get(uuid_c)
                    if rid is None:
                        continue
                    by_id[rid] = row
                return list(by_id.values())[:limit]
            return name_rows


def _warehouse_stock_column_config() -> tuple[str, str, str, str, str, str, str]:
    table = (os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_TABLE", "warehouse_stock") or "warehouse_stock").strip()
    product_c = (
        os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_PRODUCT_ID_COL", "product_id") or "product_id"
    ).strip()
    warehouse_c = (
        os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_WAREHOUSE_ID_COL", "warehouse_id") or "warehouse_id"
    ).strip()
    stock_c = (os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_STOCK_COL", "stock") or "stock").strip()
    reserved_c = (
        os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_RESERVED_COL", "reserved_stock") or "reserved_stock"
    ).strip()
    min_c = (os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_MIN_COL", "min_stock") or "min_stock").strip()
    projected_c = (
        os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_PROJECTED_COL", "stock_projected") or "stock_projected"
    ).strip()
    for c in (table, product_c, warehouse_c, stock_c, reserved_c, min_c, projected_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"identificador warehouse_stock inválido: {c!r}")
    return table, product_c, warehouse_c, stock_c, reserved_c, min_c, projected_c


def _cardex_column_config() -> tuple[str, str, str, str, str, str, str, str]:
    table = (os.environ.get("ERP_SUPABASE_CARDEX_TABLE", "cardex_report") or "cardex_report").strip()
    date_c = (os.environ.get("ERP_SUPABASE_CARDEX_DATE_COL", "fecha_hora") or "fecha_hora").strip()
    product_c = (os.environ.get("ERP_SUPABASE_CARDEX_PRODUCT_ID_COL", "product_id") or "product_id").strip()
    movement_c = (
        os.environ.get("ERP_SUPABASE_CARDEX_MOVEMENT_TYPE_COL", "tipo_movimiento") or "tipo_movimiento"
    ).strip()
    in_c = (os.environ.get("ERP_SUPABASE_CARDEX_QTY_IN_COL", "cantidad_entrada") or "cantidad_entrada").strip()
    out_c = (
        os.environ.get("ERP_SUPABASE_CARDEX_QTY_OUT_COL", "cantidad_salida") or "cantidad_salida"
    ).strip()
    sku_c = (os.environ.get("ERP_SUPABASE_CARDEX_SKU_COL", "sku") or "sku").strip()
    desc_c = (
        os.environ.get("ERP_SUPABASE_CARDEX_PRODUCT_DESC_COL", "descripcion_producto") or "descripcion_producto"
    ).strip()
    for c in (table, date_c, product_c, movement_c, in_c, out_c, sku_c, desc_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"identificador cardex inválido: {c!r}")
    return table, date_c, product_c, movement_c, in_c, out_c, sku_c, desc_c


def _coerce_limit(raw: Any, default: int = 10, cap: int = 500) -> int:
    """Groq a veces manda limit como string JSON; normalizamos."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        n = raw
    elif isinstance(raw, float):
        n = int(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return default
        try:
            n = int(s, 10)
        except ValueError:
            return default
    else:
        return default
    return max(1, min(n, cap))


def _date_range_bounds(desde: str, hasta: str) -> tuple[str, str]:
    """Si recibimos YYYY-MM-DD, cubrimos todo el día (00:00:00 a 23:59:59.999999).

    Esto evita que un campo `timestamp` con hora real nos deje afuera los
    comprobantes generados después de la medianoche del día consultado.
    """
    desde_filtro = f"{desde}T00:00:00" if len(desde) == 10 and "T" not in desde else desde
    hasta_filtro = f"{hasta}T23:59:59.999999" if len(hasta) == 10 and "T" not in hasta else hasta
    return desde_filtro, hasta_filtro


def _like_token(q: str, max_len: int = 80) -> str:
    q = q.strip()[:max_len]
    if not q:
        return "_"
    out: list[str] = []
    for c in q:
        if c.isalnum() or c in " .-_":
            out.append(c)
    s = "".join(out).strip()
    return s if s else "_"


def _orders_list_table() -> str:
    for key in (
        "ERP_SUPABASE_ORDERS_LIST_TABLE",
        "ERP_SUPABASE_SALES_ORDERS_TABLE",
        "ERP_SUPABASE_SALES_TABLE",
    ):
        t = (os.environ.get(key) or "").strip()
        if t and _safe_sql_identifier(t):
            return t
    return "sales_orders"


def _orders_list_date_column() -> str:
    for key in ("ERP_SUPABASE_ORDERS_LIST_DATE_COL", "ERP_SUPABASE_SALES_ORDER_DATE_COL"):
        c = (os.environ.get(key) or "").strip()
        if c and _safe_sql_identifier(c):
            return c
    return "order_date"


def _orders_list_select_expr() -> str:
    default = "id,order_number,order_date,total_amount,status,currency"
    s = (os.environ.get("ERP_SUPABASE_ORDERS_LIST_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_ORDERS_LIST_SELECT: columna inválida {p!r}")
    return ",".join(parts)


def _orders_list_status_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_ORDERS_STATUS_COL") or "status").strip() or "status"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_ORDERS_STATUS_COL inválida: {c!r}")
    return c


def _orders_customer_id_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_ORDERS_CUSTOMER_ID_COL") or "customer_id").strip() or "customer_id"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_ORDERS_CUSTOMER_ID_COL inválida: {c!r}")
    return c


def _orders_include_customer_data() -> bool:
    return _env_flag("ERP_SUPABASE_ORDERS_INCLUDE_CUSTOMER", True)


def _orders_seller_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_ORDERS_SELLER_COL") or "created_by").strip() or "created_by"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_ORDERS_SELLER_COL inválida: {c!r}")
    return c


def _profiles_table() -> str:
    return (os.environ.get("ERP_SUPABASE_PROFILES_TABLE") or "profiles").strip() or "profiles"


def _profiles_id_col() -> str:
    return (os.environ.get("ERP_SUPABASE_PROFILES_ID_COL") or "user_id").strip() or "user_id"


def _profiles_name_col() -> str:
    return (os.environ.get("ERP_SUPABASE_PROFILES_NAME_COL") or "full_name").strip() or "full_name"


def _invoices_seller_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICES_SELLER_COL") or "created_by").strip() or "created_by"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICES_SELLER_COL inválida: {c!r}")
    return c


def _invoices_amount_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICES_AMOUNT_COL") or "total_amount").strip() or "total_amount"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICES_AMOUNT_COL inválida: {c!r}")
    return c


def _resolve_orders_list_dates(desde_raw: Any, hasta_raw: Any) -> tuple[str, str]:
    try:
        days = int(os.environ.get("ERP_SUPABASE_ORDERS_LIST_DEFAULT_DAYS", "90") or "90")
    except ValueError:
        days = 90
    days = max(1, min(days, 3660))

    def parse_d(x: Any) -> date | None:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise ValueError(f"fecha debe ser YYYY-MM-DD, recibí: {s!r}")
        y, m, d = (int(s[0:4]), int(s[5:7]), int(s[8:10]))
        return date(y, m, d)

    d_from = parse_d(desde_raw)
    d_to = parse_d(hasta_raw)
    today = date.today()

    if d_from is None and d_to is None:
        d_to = today
        d_from = today - timedelta(days=days - 1)
    elif d_from is None:
        assert d_to is not None
        d_from = d_to - timedelta(days=days - 1)
    elif d_to is None:
        d_to = today
    if d_from > d_to:
        d_from, d_to = d_to, d_from
    return d_from.isoformat(), d_to.isoformat()


def _purchase_orders_table() -> str:
    for key in ("ERP_SUPABASE_PURCHASE_ORDERS_TABLE", "ERP_SUPABASE_PO_TABLE"):
        t = (os.environ.get(key) or "").strip()
        if t and _safe_sql_identifier(t):
            return t
    return "purchase_orders"


def _po_date_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_PO_DATE_COL") or "order_date").strip() or "order_date"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PO_DATE_COL inválida: {c!r}")
    return c


def _po_amount_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_PO_AMOUNT_COL") or "total_amount").strip() or "total_amount"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PO_AMOUNT_COL inválida: {c!r}")
    return c


def _po_status_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_PO_STATUS_COL") or "status").strip() or "status"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PO_STATUS_COL inválida: {c!r}")
    return c


def _po_list_select_expr() -> str:
    default = (
        "id,number,supplier_id,branch_id,order_date,expected_date,"
        "total_amount,tax_amount,subtotal,status,order_type,notes"
    )
    s = (os.environ.get("ERP_SUPABASE_PO_LIST_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_PO_LIST_SELECT: columna inválida {p!r}")
    return ",".join(parts)


def _po_supplier_id_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_PO_SUPPLIER_ID_COL") or "supplier_id").strip() or "supplier_id"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PO_SUPPLIER_ID_COL inválida: {c!r}")
    return c


def _po_include_supplier_data() -> bool:
    return _env_flag("ERP_SUPABASE_PO_INCLUDE_SUPPLIER", True)


def _resolve_po_list_dates(desde_raw: Any, hasta_raw: Any) -> tuple[str, str]:
    try:
        days = int(os.environ.get("ERP_SUPABASE_PO_LIST_DEFAULT_DAYS", "90") or "90")
    except ValueError:
        days = 90
    days = max(1, min(days, 3660))

    def parse_d(x: Any) -> date | None:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise ValueError(f"fecha debe ser YYYY-MM-DD, recibí: {s!r}")
        y, m, d = (int(s[0:4]), int(s[5:7]), int(s[8:10]))
        return date(y, m, d)

    d_from = parse_d(desde_raw)
    d_to = parse_d(hasta_raw)
    today = date.today()

    if d_from is None and d_to is None:
        d_to = today
        d_from = today - timedelta(days=days - 1)
    elif d_from is None:
        assert d_to is not None
        d_from = d_to - timedelta(days=days - 1)
    elif d_to is None:
        d_to = today
    if d_from > d_to:
        d_from, d_to = d_to, d_from
    return d_from.isoformat(), d_to.isoformat()


def _purchase_order_items_table() -> str:
    for key in ("ERP_SUPABASE_PURCHASE_ORDER_ITEMS_TABLE", "ERP_SUPABASE_PO_ITEMS_TABLE"):
        t = (os.environ.get(key) or "").strip()
        if t and _safe_sql_identifier(t):
            return t
    return "purchase_order_items"


def _po_items_fk_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_PO_ITEMS_FK_COL") or "purchase_order_id").strip() or "purchase_order_id"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_PO_ITEMS_FK_COL inválida: {c!r}")
    return c


def _po_items_select_expr() -> str:
    default = (
        "id,purchase_order_id,product_id,quantity,unit_price,total_price,"
        "received_quantity,unit,notes,tax_rate,tax_amount,received_weight,discount_percentage_2"
    )
    s = (os.environ.get("ERP_SUPABASE_PO_ITEMS_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_PO_ITEMS_SELECT: columna inválida {p!r}")
    return ",".join(parts)


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _sales_order_items_select_expr() -> str:
    default = "id,sales_order_id,product_id,quantity,unit_price,line_total,product_name,sku"
    s = (os.environ.get("ERP_SUPABASE_SALES_ORDER_ITEMS_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_SALES_ORDER_ITEMS_SELECT: columna inválida {p!r}")
    return ",".join(parts)


def _sales_order_number_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_ORDERS_NUMBER_COL") or "order_number").strip() or "order_number"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_ORDERS_NUMBER_COL inválida: {c!r}")
    return c


def _resolve_sales_order_id_by_number(order_number: str) -> str | None:
    client = _get_supabase()
    assert client is not None
    table = _orders_list_table()
    number_c = _sales_order_number_column()
    r = client.table(table).select("id").eq(number_c, order_number.strip()).limit(1).execute()
    rows = r.data or []
    return str(rows[0]["id"]) if rows and rows[0].get("id") else None


def _stub_list_sales_order_items(sales_order_id: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "demo-item-001",
            "sales_order_id": sales_order_id,
            "product_id": "demo-prod-1",
            "sku": "SKU-001",
            "product_name": "Producto demo A",
            "quantity": 3.0,
            "unit_price": 500.0,
            "line_total": 1500.0,
        }
    ]
    return {
        "sales_order_id": sales_order_id,
        "lineas": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _list_sales_order_items_from_supabase(sales_order_id: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    items_t, _, fk_c, *_ = _sales_items_column_config()
    select_cols = _sales_order_items_select_expr()
    lim = max(1, min(limit, 500))
    r = (
        client.table(items_t)
        .select(select_cols)
        .eq(fk_c, sales_order_id.strip())
        .limit(lim)
        .execute()
    )
    rows: list[dict[str, Any]] = r.data or []
    return {
        "sales_order_id": sales_order_id.strip(),
        "lineas": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": items_t,
    }


def _is_uuid(s: str) -> bool:
    return bool(_UUID_RE.fullmatch(s.strip()))


# ── Facturas: config ────────────────────────────────────────────────────────

def _invoices_table() -> str:
    t = (os.environ.get("ERP_SUPABASE_INVOICES_TABLE") or "customer_invoices").strip() or "customer_invoices"
    if not _safe_sql_identifier(t):
        raise ValueError(f"ERP_SUPABASE_INVOICES_TABLE inválida: {t!r}")
    return t


def _invoices_date_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICES_DATE_COL") or "issue_date").strip() or "issue_date"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICES_DATE_COL inválida: {c!r}")
    return c


def _invoices_customer_id_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICES_CUSTOMER_ID_COL") or "customer_id").strip() or "customer_id"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICES_CUSTOMER_ID_COL inválida: {c!r}")
    return c


def _invoice_number_column() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICES_NUMBER_COL") or "invoice_number").strip() or "invoice_number"
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICES_NUMBER_COL inválida: {c!r}")
    return c


def _invoices_select_expr() -> str:
    default = "id,invoice_number,issue_date,due_date,total_amount,paid_amount,remaining_amount,status,currency,receipt_type,customer_id"
    s = (os.environ.get("ERP_SUPABASE_INVOICES_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_INVOICES_SELECT: columna inválida {p!r}")
    return ",".join(parts)


def _invoice_items_table() -> str:
    t = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_TABLE") or "customer_invoice_items").strip() or "customer_invoice_items"
    if not _safe_sql_identifier(t):
        raise ValueError(f"ERP_SUPABASE_INVOICE_ITEMS_TABLE inválida: {t!r}")
    return t


def _invoice_items_select_expr() -> str:
    default = "id,customer_invoice_id,product_id,quantity,unit_price,discount_percentage,line_total,tax_rate,description"
    s = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_INVOICE_ITEMS_SELECT: columna inválida {p!r}")
    return ",".join(parts)


def _invoice_items_product_id_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PRODUCT_ID_COL", "product_id") or "product_id").strip()
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICE_ITEMS_PRODUCT_ID_COL inválida: {c!r}")
    return c


def _invoice_items_qty_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_QTY_COL", "quantity") or "quantity").strip()
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICE_ITEMS_QTY_COL inválida: {c!r}")
    return c


def _invoice_items_fk_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_FK_COL", "customer_invoice_id") or "customer_invoice_id").strip()
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICE_ITEMS_FK_COL inválida: {c!r}")
    return c


def _invoice_items_amount_col() -> str:
    c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_AMOUNT_COL", "line_total") or "line_total").strip()
    if not _safe_sql_identifier(c):
        raise ValueError(f"ERP_SUPABASE_INVOICE_ITEMS_AMOUNT_COL inválida: {c!r}")
    return c


def _payments_table() -> str:
    t = (os.environ.get("ERP_SUPABASE_PAYMENTS_TABLE") or "customer_payments").strip() or "customer_payments"
    if not _safe_sql_identifier(t):
        raise ValueError(f"ERP_SUPABASE_PAYMENTS_TABLE inválida: {t!r}")
    return t


def _payments_select_expr() -> str:
    default = "id,customer_invoice_id,payment_date,amount,payment_method,reference_number,status,notes"
    s = (os.environ.get("ERP_SUPABASE_PAYMENTS_SELECT") or default).strip() or default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if not _safe_sql_identifier(p):
            raise ValueError(f"ERP_SUPABASE_PAYMENTS_SELECT: columna inválida {p!r}")
    return ",".join(parts)


def _resolve_invoice_id_by_number(invoice_number: str) -> str | None:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    number_c = _invoice_number_column()
    r = client.table(table).select("id").filter(number_c, "eq", invoice_number.strip()).limit(1).execute()
    rows = r.data or []
    return str(rows[0]["id"]) if rows and rows[0].get("id") else None


# ── Facturas: stubs ─────────────────────────────────────────────────────────

def _stub_list_customer_invoices(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "00000000-0000-0000-0000-000000000010",
            "invoice_number": "FC-DEMO-001",
            "issue_date": desde,
            "due_date": hasta,
            "total_amount": 75000.0,
            "paid_amount": 50000.0,
            "remaining_amount": 25000.0,
            "status": "partial",
            "currency": "ARS",
            "receipt_type": "factura",
        }
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "facturas": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para datos reales",
    }


def _stub_get_invoice_summary(desde: str, hasta: str) -> dict[str, Any]:
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_facturado": 500000.0,
        "total_cobrado": 350000.0,
        "saldo_pendiente": 150000.0,
        "cantidad_facturas": 12,
        "por_estado": {"paid": 5, "partial": 4, "pending": 2, "overdue": 1},
        "fuente": "stub",
    }


def _stub_list_customer_invoice_items(invoice_id: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "demo-inv-item-001",
            "customer_invoice_id": invoice_id,
            "product_id": "demo-prod-1",
            "quantity": 2.0,
            "unit_price": 15000.0,
            "line_total": 30000.0,
            "tax_rate": 21.0,
            "description": "Producto demo",
        }
    ]
    return {
        "invoice_id": invoice_id,
        "lineas": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "fuente": "stub",
    }


def _stub_list_customer_payments(limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "demo-pay-001",
            "customer_invoice_id": "00000000-0000-0000-0000-000000000010",
            "payment_date": "2026-01-15",
            "amount": 50000.0,
            "payment_method": "bank_transfer",
            "reference_number": "TRF-DEMO-001",
            "status": "pending_validation",
        }
    ]
    return {
        "pagos": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "fuente": "stub",
    }


# ── Facturas: Supabase handlers ─────────────────────────────────────────────

def _list_customer_invoices_from_supabase(
    desde: str, hasta: str, limit: int,
    filter_customer_id: str | None = None,
    filter_status: str | None = None,
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    date_c = _invoices_date_column()
    customer_id_c = _invoices_customer_id_column()
    select_cols = _invoices_select_expr()
    lim = max(1, min(limit, 200))
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)
    # Statuses/types excluidos cuando no hay filtro explícito de estado (igual que el front)
    excluded_statuses = ["cancelled", "voided", "converted", "draft"]
    excluded_types = {"nota_pedido", "np"}
    # Asegurar que los campos de filtro de validInvoices estén en el select
    extra_cols = "invoice_type,warehouse_id,sales_order_id,payment_condition"
    full_select = select_cols if all(c in select_cols for c in ["warehouse_id", "sales_order_id"]) \
        else f"{select_cols},{extra_cols}"
    # Base query (reutilizada para count y para datos)
    def _base_q():
        bq = client.table(table).select(full_select).gte(date_c, desde_filtro).lte(date_c, hasta_filtro)
        if filter_customer_id:
            bq = bq.eq(customer_id_c, filter_customer_id)
        if filter_status:
            bq = bq.eq("status", filter_status)
        else:
            bq = bq.not_.in_("status", excluded_statuses)
        return bq
    # COUNT exacto para saber el total real en la base de datos
    total_en_bd: int | None = None
    try:
        count_r = client.table(table).select("id", count="exact").gte(date_c, desde_filtro).lte(date_c, hasta_filtro)
        if filter_customer_id:
            count_r = count_r.eq(customer_id_c, filter_customer_id)
        if filter_status:
            count_r = count_r.eq("status", filter_status)
        else:
            count_r = count_r.not_.in_("status", excluded_statuses)
        count_r = count_r.execute()
        total_en_bd = count_r.count
    except Exception:
        pass
    r = _base_q().order(date_c, desc=True).limit(lim).execute()
    rows: list[dict[str, Any]] = r.data or []
    # Filtros cliente (igual que validInvoices del front) cuando no hay filtro explícito de estado
    if not filter_status:
        rows = [
            x for x in rows
            if (x.get("invoice_type") or "").lower() not in excluded_types
            # Al menos uno de estos campos debe estar presente
            and (x.get("warehouse_id") or x.get("sales_order_id") or x.get("payment_condition"))
        ]
    rows = _attach_customers_to_orders(rows, customer_id_c)
    hay_mas = (total_en_bd is not None and total_en_bd > len(rows)) or (total_en_bd is None and len(rows) >= lim)
    result: dict[str, Any] = {
        "periodo": {"desde": desde, "hasta": hasta},
        "facturas": rows,
        "cantidad_devuelta": len(rows),
        "limite_aplicado": lim,
        "hay_mas": hay_mas,
        "fuente": "supabase",
        "tabla": table,
    }
    if total_en_bd is not None:
        result["total_en_bd"] = total_en_bd
    return result


def _get_invoice_summary_from_supabase(
    desde: str, hasta: str,
    filter_customer_id: str | None = None,
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    date_c = _invoices_date_column()
    customer_id_c = _invoices_customer_id_column()
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)

    # Paso 1: IDs que son destino de conversión (evitar doble conteo remito→factura)
    excluded_ids: set[str] = set()
    try:
        excl_start = 0
        pg = 1000
        while True:
            excl_r = (
                client.table(table)
                .select("converted_to_invoice_id")
                .not_.is_("converted_to_invoice_id", "null")
                .order("id")
                .range(excl_start, excl_start + pg - 1)
                .execute()
            )
            excl_rows = excl_r.data or []
            for row in excl_rows:
                if row.get("converted_to_invoice_id"):
                    excluded_ids.add(row["converted_to_invoice_id"])
            if len(excl_rows) < pg:
                break
            excl_start += pg
            if excl_start > 500_000:
                break
    except Exception:
        pass

    # Paso 2: paginar facturas con filtros de fecha y estado (igual que el front: validInvoices)
    excluded_statuses = ["cancelled", "voided", "converted", "draft"]
    excluded_types = {"nota_pedido", "np"}
    pg = 1000
    start = 0
    rows: list[dict[str, Any]] = []
    while True:
        q = (
            client.table(table)
            .select("id,total_amount,paid_amount,remaining_amount,status,invoice_type,warehouse_id,sales_order_id,payment_condition")
            .gte(date_c, desde_filtro)
            .lte(date_c, hasta_filtro)
            .not_.in_("status", excluded_statuses)
            .order("id")
            .range(start, start + pg - 1)
        )
        if filter_customer_id:
            q = q.eq(customer_id_c, filter_customer_id)
        r = q.execute()
        batch: list[dict[str, Any]] = r.data or []
        for x in batch:
            if x.get("id") in excluded_ids:
                continue
            if (x.get("invoice_type") or "").lower() in excluded_types:
                continue
            # Al menos uno de estos campos debe estar presente (igual que validInvoices del front)
            if not (x.get("warehouse_id") or x.get("sales_order_id") or x.get("payment_condition")):
                continue
            rows.append(x)
        if len(batch) < pg:
            break
        start += pg
        if start > 500_000:
            break

    total_facturado = sum(float(x.get("total_amount") or 0) for x in rows)
    total_cobrado = sum(float(x.get("paid_amount") or 0) for x in rows)
    saldo_pendiente = sum(float(x.get("remaining_amount") or 0) for x in rows)
    por_estado: dict[str, int] = {}
    for x in rows:
        st = str(x.get("status") or "unknown")
        por_estado[st] = por_estado.get(st, 0) + 1
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_facturado": round(total_facturado, 2),
        "total_cobrado": round(total_cobrado, 2),
        "saldo_pendiente": round(saldo_pendiente, 2),
        "cantidad_facturas": len(rows),
        "por_estado": por_estado,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_get_profit_margin(desde: str, hasta: str) -> dict[str, Any]:
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_ventas": 500000.0,
        "total_costo_mercaderia": 350000.0,
        "utilidad": 150000.0,
        "markup_pct": 42.9,
        "pct_utilidad_sobre_ventas": 30.0,
        "cantidad_facturas": 12,
        "items_procesados": 48,
        "items_con_costo_registrado": 48,
        "items_sin_costo": 0,
        "fuente": "stub",
        "nota_metodologia": "Costo de mercadería c/ IVA. Markup = Utilidad / Costo × 100. % Utilidad = Utilidad / Venta × 100.",
    }


def _get_profit_margin_from_supabase(
    desde: str, hasta: str,
    filter_customer_id: str | None = None,
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    date_c = _invoices_date_column()
    customer_id_c = _invoices_customer_id_column()
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)

    # Paso 1: IDs de facturas convertidas a excluir (evitar doble conteo)
    excluded_ids: set[str] = set()
    try:
        excl_start = 0
        pg = 1000
        while True:
            excl_r = (
                client.table(table)
                .select("converted_to_invoice_id")
                .not_.is_("converted_to_invoice_id", "null")
                .order("id")
                .range(excl_start, excl_start + pg - 1)
                .execute()
            )
            excl_rows = excl_r.data or []
            for row in excl_rows:
                if row.get("converted_to_invoice_id"):
                    excluded_ids.add(row["converted_to_invoice_id"])
            if len(excl_rows) < pg:
                break
            excl_start += pg
            if excl_start > 500_000:
                break
    except Exception:
        pass

    # Paso 2: Obtener facturas válidas con total_amount (igual que get_invoice_summary)
    excluded_statuses = ["cancelled", "voided", "converted", "draft"]
    excluded_types = {"nota_pedido", "np"}
    pg = 1000
    start = 0
    invoice_rows: list[dict[str, Any]] = []
    while True:
        q = (
            client.table(table)
            .select("id,total_amount,invoice_type,warehouse_id,sales_order_id,payment_condition")
            .gte(date_c, desde_filtro)
            .lte(date_c, hasta_filtro)
            .not_.in_("status", excluded_statuses)
            .order("id")
            .range(start, start + pg - 1)
        )
        if filter_customer_id:
            q = q.eq(customer_id_c, filter_customer_id)
        r = q.execute()
        batch: list[dict[str, Any]] = r.data or []
        for x in batch:
            if x.get("id") in excluded_ids:
                continue
            if (x.get("invoice_type") or "").lower() in excluded_types:
                continue
            if not (x.get("warehouse_id") or x.get("sales_order_id") or x.get("payment_condition")):
                continue
            invoice_rows.append(x)
        if len(batch) < pg:
            break
        start += pg
        if start > 500_000:
            break

    if not invoice_rows:
        return {
            "periodo": {"desde": desde, "hasta": hasta},
            "total_ventas": 0.0,
            "total_costo_mercaderia": 0.0,
            "utilidad": 0.0,
            "markup_pct": 0.0,
            "pct_utilidad_sobre_ventas": 0.0,
            "cantidad_facturas": 0,
            "fuente": "supabase",
        }

    total_ventas = sum(_to_float(x.get("total_amount")) for x in invoice_rows)
    invoice_ids = [x["id"] for x in invoice_rows if x.get("id")]

    # Paso 3: Fetch ítems de factura con columnas de costo
    items_table = _invoice_items_table()
    fk_c = _invoice_items_fk_col()
    product_id_c = _invoice_items_product_id_col()
    qty_c = _invoice_items_qty_col()

    purchase_cost_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_COST_COL") or "purchase_cost").strip()
    purchase_cost_tax_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_COST_INCLUDES_TAX_COL") or "purchase_cost_includes_tax").strip()
    purchase_vat_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_VAT_RATE_COL") or "purchase_vat_rate").strip()

    for c in (purchase_cost_c, purchase_cost_tax_c, purchase_vat_c):
        if not _safe_sql_identifier(c):
            raise ValueError(f"columna de costo de ítem inválida: {c!r}")

    try:
        chunk_sz = int(os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_INVOICE_ID_CHUNK", "80") or "80")
    except ValueError:
        chunk_sz = 80
    chunk_sz = max(1, min(chunk_sz, 200))

    item_select = f"{fk_c},{product_id_c},{qty_c},{purchase_cost_c},{purchase_cost_tax_c},{purchase_vat_c}"
    all_items: list[dict[str, Any]] = []
    product_ids_needed: set[Any] = set()

    for i in range(0, len(invoice_ids), chunk_sz):
        chunk = invoice_ids[i : i + chunk_sz]
        start_inner = 0
        while True:
            try:
                r_items = (
                    client.table(items_table)
                    .select(item_select)
                    .in_(fk_c, chunk)
                    .range(start_inner, start_inner + 999)
                    .execute()
                )
            except Exception:
                # Fallback si alguna columna de costo no existe en este deploy
                r_items = (
                    client.table(items_table)
                    .select(f"{fk_c},{product_id_c},{qty_c},{purchase_cost_c}")
                    .in_(fk_c, chunk)
                    .range(start_inner, start_inner + 999)
                    .execute()
                )
            rows_chunk: list[dict[str, Any]] = r_items.data or []
            for row in rows_chunk:
                all_items.append(row)
                pid = row.get(product_id_c)
                if pid:
                    product_ids_needed.add(pid)
            if len(rows_chunk) < 1000:
                break
            start_inner += 1000
            if start_inner > 500_000:
                break

    # Paso 4: Obtener costo del producto como fallback
    products_cost: dict[Any, dict[str, Any]] = {}
    if product_ids_needed:
        _, _, _, uuid_c = _product_table_columns()
        if not uuid_c:
            uuid_c = "id"
        prod_cost_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_PRICE") or "cost_price").strip()
        prod_cost_tax_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_INCLUDES_TAX") or "cost_price_includes_tax").strip()
        prod_vat_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_VAT_RATE") or "vat_rate").strip()
        prod_ids_list = list(product_ids_needed)
        try:
            prod_r = (
                client.table(_products_table())
                .select(f"{uuid_c},{prod_cost_col},{prod_cost_tax_col},{prod_vat_col}")
                .in_(uuid_c, prod_ids_list)
                .execute()
            )
            for p in (prod_r.data or []):
                pid = p.get(uuid_c)
                if pid:
                    products_cost[pid] = {
                        "cost_price": _to_float(p.get(prod_cost_col)),
                        "cost_includes_tax": bool(p.get(prod_cost_tax_col)),
                        "vat_rate": _to_float(p.get(prod_vat_col)),
                    }
        except Exception:
            pass

    # Paso 5: Calcular costo total con IVA (igual que useSalesReports.ts)
    total_costo = 0.0
    items_con_costo = 0
    items_sin_costo = 0

    for item in all_items:
        qty = _to_float(item.get(qty_c))
        raw_cost = item.get(purchase_cost_c)

        if raw_cost is not None and raw_cost != "":
            base_cost = _to_float(raw_cost)
            includes_tax_raw = item.get(purchase_cost_tax_c)
            includes_tax = bool(includes_tax_raw) if includes_tax_raw is not None else False
            vat_rate = _to_float(item.get(purchase_vat_c))
            items_con_costo += 1
        else:
            pid = item.get(product_id_c)
            prod_info = products_cost.get(pid, {})
            base_cost = prod_info.get("cost_price", 0.0)
            includes_tax = prod_info.get("cost_includes_tax", False)
            vat_rate = prod_info.get("vat_rate", 0.0)
            if base_cost > 0:
                items_con_costo += 1
            else:
                items_sin_costo += 1

        tax_mult = 1.0 if includes_tax else (1.0 + vat_rate / 100.0)
        total_costo += qty * base_cost * tax_mult

    utilidad = total_ventas - total_costo
    markup_pct = round((utilidad / total_costo) * 100, 2) if total_costo > 0 else 0.0
    pct_utilidad_ventas = round((utilidad / total_ventas) * 100, 2) if total_ventas > 0 else 0.0

    result: dict[str, Any] = {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_ventas": round(total_ventas, 2),
        "total_costo_mercaderia": round(total_costo, 2),
        "utilidad": round(utilidad, 2),
        "markup_pct": markup_pct,
        "pct_utilidad_sobre_ventas": pct_utilidad_ventas,
        "cantidad_facturas": len(invoice_rows),
        "items_procesados": len(all_items),
        "items_con_costo_registrado": items_con_costo,
        "items_sin_costo": items_sin_costo,
        "fuente": "supabase",
        "nota_metodologia": (
            "Costo de mercadería c/ IVA (igual que el reporte del ERP). "
            "Markup = Utilidad / Costo × 100. "
            "% Utilidad = Utilidad / Venta × 100."
        ),
    }
    if items_sin_costo > 0:
        result["advertencia"] = (
            f"{items_sin_costo} ítem(s) sin costo registrado ni cost_price en el producto "
            "(no incluidos en el costo total; el margen puede estar sobreestimado)."
        )
    return result


def _list_customer_invoice_items_from_supabase(invoice_id: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoice_items_table()
    select_cols = _invoice_items_select_expr()
    lim = max(1, min(limit, 500))
    r = client.table(table).select(select_cols).eq("customer_invoice_id", invoice_id.strip()).limit(lim).execute()
    rows: list[dict[str, Any]] = r.data or []
    return {
        "invoice_id": invoice_id.strip(),
        "lineas": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_get_product_sales_units(query: str, desde: str, hasta: str) -> dict[str, Any]:
    return {
        "query": query,
        "periodo": {"desde": desde, "hasta": hasta},
        "productos": [
            {
                "nombre": f"Producto demo ({query})",
                "sku_o_codigo": "DEMO-001",
                "sale_unit": "KG",
                "cantidad_vendida": 42.0,
                "kg_vendidos": 42.0,
                "monto_vendido": 21000.0,
                "costo_mercaderia": 14700.0,
                "utilidad": 6300.0,
                "markup_pct": 42.9,
                "pct_utilidad_ventas": 30.0,
                "cantidad_ventas": 7,
            }
        ],
        "cantidad_productos_encontrados": 1,
        "fuente": "stub",
    }


def _get_product_sales_units_from_supabase(query: str, desde: str, hasta: str) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None

    # 1. Resolver producto(s) por nombre/SKU (búsqueda parcial ilike)
    products = _resolve_products_by_query(query, limit=10)
    if not products:
        return {
            "query": query,
            "periodo": {"desde": desde, "hasta": hasta},
            "productos": [],
            "cantidad_productos_encontrados": 0,
            "mensaje": f"No se encontraron productos con '{query}'",
            "fuente": "supabase",
        }

    code_c, name_c, _, uuid_c = _product_table_columns()
    if not uuid_c:
        raise ValueError("ERP_SUPABASE_PRODUCT_COL_UUID es obligatorio para consultar ventas por producto")

    # Construir mapa product_uuid → info del producto
    product_map: dict[Any, dict[str, Any]] = {}
    for p in products:
        pid = p.get("id") or p.get(uuid_c)
        if pid:
            product_map[pid] = {
                "nombre": p.get("nombre") or p.get(name_c),
                "sku_o_codigo": p.get("sku_o_codigo") or p.get(code_c),
                "sale_unit": "",
                "unit_weight": 0.0,
            }

    if not product_map:
        return {
            "query": query,
            "periodo": {"desde": desde, "hasta": hasta},
            "productos": [],
            "cantidad_productos_encontrados": len(products),
            "mensaje": "Productos encontrados pero sin UUID (revisar ERP_SUPABASE_PRODUCT_COL_UUID)",
            "fuente": "supabase",
        }

    product_ids = list(product_map.keys())

    # 2. Obtener sale_unit, unit_weight y columnas de costo del producto (fallback)
    sale_unit_c = _product_col_sale_unit()
    unit_weight_c = _product_col_unit_weight()
    prod_cost_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_PRICE") or "cost_price").strip()
    prod_cost_tax_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_INCLUDES_TAX") or "cost_price_includes_tax").strip()
    prod_vat_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_VAT_RATE") or "vat_rate").strip()
    try:
        wp_r = (
            client.table(_products_table())
            .select(f"{uuid_c},{sale_unit_c},{unit_weight_c},{prod_cost_col},{prod_cost_tax_col},{prod_vat_col}")
            .in_(uuid_c, product_ids)
            .execute()
        )
        for wp in (wp_r.data or []):
            pid = wp.get(uuid_c)
            if pid in product_map:
                product_map[pid]["sale_unit"] = (wp.get(sale_unit_c) or "").upper()
                product_map[pid]["unit_weight"] = _to_float(wp.get(unit_weight_c))
                product_map[pid]["cost_price"] = _to_float(wp.get(prod_cost_col))
                product_map[pid]["cost_includes_tax"] = bool(wp.get(prod_cost_tax_col))
                product_map[pid]["vat_rate"] = _to_float(wp.get(prod_vat_col))
    except Exception:
        # Si las columnas no existen en este deploy, continuar sin info de peso/costo
        try:
            wp_r = (
                client.table(_products_table())
                .select(f"{uuid_c},{sale_unit_c},{unit_weight_c}")
                .in_(uuid_c, product_ids)
                .execute()
            )
            for wp in (wp_r.data or []):
                pid = wp.get(uuid_c)
                if pid in product_map:
                    product_map[pid]["sale_unit"] = (wp.get(sale_unit_c) or "").upper()
                    product_map[pid]["unit_weight"] = _to_float(wp.get(unit_weight_c))
        except Exception:
            pass

    # 3. Obtener IDs de facturas válidas en el rango de fechas
    invoices_table = _invoices_table()
    date_c = _invoices_date_column()
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)
    excluded_statuses = ["cancelled", "voided", "converted", "draft"]
    excluded_types = {"nota_pedido", "np"}

    page = 1000
    invoice_ids: list[Any] = []
    start = 0
    while True:
        r = (
            client.table(invoices_table)
            .select("id,invoice_type,warehouse_id,sales_order_id,payment_condition")
            .gte(date_c, desde_filtro)
            .lte(date_c, hasta_filtro)
            .not_.in_("status", excluded_statuses)
            .range(start, start + page - 1)
            .execute()
        )
        rows = r.data or []
        for row in rows:
            if (row.get("invoice_type") or "").lower() not in excluded_types:
                if row.get("warehouse_id") or row.get("sales_order_id") or row.get("payment_condition"):
                    if row.get("id"):
                        invoice_ids.append(row["id"])
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    invoice_ids = list(dict.fromkeys(invoice_ids))
    if not invoice_ids:
        return {
            "query": query,
            "periodo": {"desde": desde, "hasta": hasta},
            "productos": [
                {
                    "nombre": info["nombre"],
                    "sku_o_codigo": info["sku_o_codigo"],
                    "sale_unit": info["sale_unit"],
                    "cantidad_vendida": 0.0,
                    "kg_vendidos": 0.0,
                    "monto_vendido": 0.0,
                    "cantidad_ventas": 0,
                }
                for info in product_map.values()
            ],
            "cantidad_productos_encontrados": len(product_map),
            "fuente": "supabase",
        }

    # 4. Agregar unidades, kg, monto, costo y conteo de ventas desde customer_invoice_items
    items_table = _invoice_items_table()
    fk_c = _invoice_items_fk_col()
    product_id_c = _invoice_items_product_id_col()
    qty_c = _invoice_items_qty_col()
    amount_c = _invoice_items_amount_col()

    purchase_cost_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_COST_COL") or "purchase_cost").strip()
    purchase_cost_tax_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_COST_INCLUDES_TAX_COL") or "purchase_cost_includes_tax").strip()
    purchase_vat_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_VAT_RATE_COL") or "purchase_vat_rate").strip()
    # tax_amount: IVA por línea — igual que el ERP: Venta Final = line_total + tax_amount
    tax_amount_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_TAX_AMOUNT_COL") or "tax_amount").strip()

    try:
        chunk_sz = int(os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_INVOICE_ID_CHUNK", "80") or "80")
    except ValueError:
        chunk_sz = 80
    chunk_sz = max(1, min(chunk_sz, 200))

    agg: dict[Any, dict[str, Any]] = {
        pid: {"cantidad_vendida": 0.0, "kg_vendidos": 0.0, "monto_vendido": 0.0, "costo_total": 0.0, "_invoice_ids": set()}
        for pid in product_ids
    }

    item_select_with_cost = f"{fk_c},{product_id_c},{qty_c},{amount_c},{tax_amount_c},{purchase_cost_c},{purchase_cost_tax_c},{purchase_vat_c}"
    item_select_base = f"{fk_c},{product_id_c},{qty_c},{amount_c},{tax_amount_c}"

    for i in range(0, len(invoice_ids), chunk_sz):
        chunk = invoice_ids[i : i + chunk_sz]
        start_inner = 0
        while True:
            try:
                r = (
                    client.table(items_table)
                    .select(item_select_with_cost)
                    .in_(fk_c, chunk)
                    .in_(product_id_c, product_ids)
                    .range(start_inner, start_inner + page - 1)
                    .execute()
                )
            except Exception:
                # Fallback sin columnas de costo de ítem
                r = (
                    client.table(items_table)
                    .select(item_select_base)
                    .in_(fk_c, chunk)
                    .in_(product_id_c, product_ids)
                    .range(start_inner, start_inner + page - 1)
                    .execute()
                )
            rows = r.data or []
            for row in rows:
                pid = row.get(product_id_c)
                if pid not in agg:
                    continue
                raw_qty = _to_float(row.get(qty_c))
                agg[pid]["cantidad_vendida"] += raw_qty
                # Venta Final c/ IVA = line_total + tax_amount (igual que el ERP)
                agg[pid]["monto_vendido"] += _to_float(row.get(amount_c)) + _to_float(row.get(tax_amount_c))
                # Calcular kg
                info = product_map[pid]
                sale_unit = info.get("sale_unit", "")
                unit_weight = info.get("unit_weight", 0.0)
                if "KG" in sale_unit or "LBS" in sale_unit:
                    agg[pid]["kg_vendidos"] += raw_qty
                elif unit_weight:
                    agg[pid]["kg_vendidos"] += raw_qty * unit_weight
                # Calcular costo con IVA (igual que useSalesReports.ts)
                raw_cost = row.get(purchase_cost_c)
                if raw_cost is not None and raw_cost != "":
                    base_cost = _to_float(raw_cost)
                    includes_tax_raw = row.get(purchase_cost_tax_c)
                    includes_tax = bool(includes_tax_raw) if includes_tax_raw is not None else False
                    vat_rate = _to_float(row.get(purchase_vat_c))
                else:
                    # Fallback al cost_price del producto
                    base_cost = info.get("cost_price", 0.0)
                    includes_tax = info.get("cost_includes_tax", False)
                    vat_rate = info.get("vat_rate", 0.0)
                tax_mult = 1.0 if includes_tax else (1.0 + vat_rate / 100.0)
                agg[pid]["costo_total"] += raw_qty * base_cost * tax_mult
                # Trackear factura distinta
                inv_id = row.get(fk_c)
                if inv_id:
                    agg[pid]["_invoice_ids"].add(inv_id)
            if len(rows) < page:
                break
            start_inner += page
            if start_inner > 500_000:
                break

    # Construir resultado final con costo y margen
    result_list = []
    for pid in product_ids:
        monto = agg[pid]["monto_vendido"]
        costo = round(agg[pid]["costo_total"], 2)
        utilidad = round(monto - costo, 2)
        markup_pct = round((utilidad / costo) * 100, 2) if costo > 0 else 0.0
        pct_utilidad = round((utilidad / monto) * 100, 2) if monto > 0 else 0.0
        result_list.append({
            "nombre": product_map[pid]["nombre"],
            "sku_o_codigo": product_map[pid]["sku_o_codigo"],
            "sale_unit": product_map[pid]["sale_unit"],
            "cantidad_vendida": agg[pid]["cantidad_vendida"],
            "kg_vendidos": round(agg[pid]["kg_vendidos"], 3),
            "monto_vendido": round(monto, 2),
            "costo_mercaderia": costo,
            "utilidad": utilidad,
            "markup_pct": markup_pct,
            "pct_utilidad_ventas": pct_utilidad,
            "cantidad_ventas": len(agg[pid]["_invoice_ids"]),
        })
    result_list.sort(key=lambda x: x["cantidad_vendida"], reverse=True)

    return {
        "query": query,
        "periodo": {"desde": desde, "hasta": hasta},
        "productos": result_list,
        "cantidad_productos_encontrados": len(result_list),
        "fuente": "supabase",
        "tabla_items": items_table,
        "nota_costo": "Costo de mercadería c/ IVA (desde purchase_cost del ítem o cost_price del producto). Markup = Utilidad / Costo × 100.",
    }


def _list_customer_payments_from_supabase(
    limit: int,
    filter_invoice_id: str | None = None,
    filter_customer_id: str | None = None,
    desde: str | None = None,
    hasta: str | None = None,
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _payments_table()
    select_cols = _payments_select_expr()
    lim = max(1, min(limit, 200))
    q = client.table(table).select(select_cols)
    if filter_invoice_id:
        q = q.eq("customer_invoice_id", filter_invoice_id)
    if desde:
        q = q.gte("payment_date", desde)
    if hasta:
        q = q.lte("payment_date", hasta)
    if filter_customer_id and not filter_invoice_id:
        inv_table = _invoices_table()
        inv_r = (
            client.table(inv_table)
            .select("id")
            .eq(_invoices_customer_id_column(), filter_customer_id)
            .limit(500)
            .execute()
        )
        inv_ids = [x["id"] for x in (inv_r.data or []) if x.get("id")]
        if not inv_ids:
            return {"pagos": [], "cantidad_devuelta": 0, "fuente": "supabase"}
        q = q.in_("customer_invoice_id", inv_ids)
    r = q.order("payment_date", desc=True).limit(lim).execute()
    rows: list[dict[str, Any]] = r.data or []
    return {
        "pagos": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_list_sales_orders(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "order_number": "OV-DEMO-001",
            "order_date": desde,
            "total_amount": 1500.0,
            "status": "complete",
            "currency": "ARS",
        }
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "ordenes": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _resolve_customer_id_by_name(name: str) -> str | None:
    client = _get_supabase()
    assert client is not None
    table, id_c, name_c, *_ = _customer_column_config()
    tok = _like_token(name)
    r = client.table(table).select(id_c).filter(name_c, "ilike", f"%{tok}%").limit(1).execute()
    rows = r.data or []
    return str(rows[0][id_c]) if rows and rows[0].get(id_c) else None


def _customer_balance_from_supabase(customer: str) -> dict[str, Any]:
    """Consulta customer_balances_view para un cliente dado (nombre, CUIT o UUID)."""
    client = _get_supabase()
    assert client is not None

    import re as _re
    _UUID_RE = _re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", _re.IGNORECASE)

    # Resolver customer_id
    if _UUID_RE.match(customer.strip()):
        customer_id = customer.strip()
        customer_name: str | None = None
    else:
        table, id_c, name_c, tax_c, *_ = _customer_column_config()
        tok = _like_token(customer)
        pat = f"%{tok}%"
        r = (
            client.table(table)
            .select(f"{id_c},{name_c},{tax_c}")
            .or_(f"{name_c}.ilike.{pat},{tax_c}.ilike.{pat}")
            .limit(1)
            .execute()
        )
        rows = r.data or []
        if not rows:
            return {"error": f"No se encontró ningún cliente que coincida con '{customer}'"}
        customer_id = str(rows[0][id_c])
        customer_name = str(rows[0].get(name_c) or "")

    # Consultar vista de saldos
    r2 = (
        client.from_("customer_balances_view")  # type: ignore[arg-type]
        .select("customer_id,total_debt,total_debt_bn1,total_debt_bn2,overdue_debt,current_debt,pending_invoices_count")
        .eq("customer_id", customer_id)
        .limit(1)
        .execute()
    )
    rows2 = r2.data or []
    if not rows2:
        return {
            "customer_id": customer_id,
            "nombre": customer_name,
            "total_debt": 0.0,
            "situacion": "sin_movimientos",
            "mensaje": "El cliente no tiene movimientos registrados en cuenta corriente.",
        }

    row = rows2[0]
    total_debt = float(row.get("total_debt") or 0)
    situacion = "deuda" if total_debt > 0 else ("saldo_a_favor" if total_debt < 0 else "saldado")
    return {
        "customer_id": customer_id,
        "nombre": customer_name,
        "total_debt": round(total_debt, 2),
        "total_debt_bn1": round(float(row.get("total_debt_bn1") or 0), 2),
        "total_debt_bn2": round(float(row.get("total_debt_bn2") or 0), 2),
        "overdue_debt": round(float(row.get("overdue_debt") or 0), 2),
        "current_debt": round(float(row.get("current_debt") or 0), 2),
        "pending_invoices_count": int(row.get("pending_invoices_count") or 0),
        "situacion": situacion,
        "fuente": "customer_balances_view",
    }


def _list_sales_orders_from_supabase(
    desde: str, hasta: str, limit: int, filter_customer_id: str | None = None
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _orders_list_table()
    date_c = _orders_list_date_column()
    select_cols = _orders_list_select_expr()
    customer_id_col = _orders_customer_id_column()
    select_parts = [p.strip() for p in select_cols.split(",") if p.strip()]
    if _orders_include_customer_data() and customer_id_col not in select_parts:
        select_parts.append(customer_id_col)
    select_cols = ",".join(select_parts)
    lim = max(1, min(limit, 200))
    # COUNT exacto para saber el total real en la base de datos
    total_en_bd: int | None = None
    try:
        count_q = client.table(table).select("id", count="exact").gte(date_c, desde).lte(date_c, hasta)
        if filter_customer_id:
            count_q = count_q.eq(customer_id_col, filter_customer_id)
        count_r = count_q.execute()
        total_en_bd = count_r.count
    except Exception:
        pass
    q = (
        client.table(table)
        .select(select_cols)
        .gte(date_c, desde)
        .lte(date_c, hasta)
    )
    if filter_customer_id:
        q = q.eq(customer_id_col, filter_customer_id)
    r = q.order(date_c, desc=True).limit(lim).execute()
    rows: list[dict[str, Any]] = r.data or []
    if _orders_include_customer_data():
        rows = _attach_customers_to_orders(rows, customer_id_col)
    amount_col = (os.environ.get("ERP_SUPABASE_ORDERS_LIST_SELECT") or "total_amount")
    # Determinar columna de importe del listado
    amount_cols = [p.strip() for p in (os.environ.get("ERP_SUPABASE_ORDERS_LIST_SELECT") or "total_amount").split(",") if p.strip()]
    _amount_c = next((c for c in amount_cols if "amount" in c.lower() or "total" in c.lower()), None)
    total_monto_devuelto = None
    if _amount_c:
        try:
            total_monto_devuelto = round(sum(float(row.get(_amount_c) or 0) for row in rows), 2)
        except Exception:
            pass
    hay_mas = (total_en_bd is not None and total_en_bd > len(rows)) or (total_en_bd is None and len(rows) >= lim)
    result: dict[str, Any] = {
        "periodo": {"desde": desde, "hasta": hasta},
        "ordenes": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "hay_mas": hay_mas,
        "fuente": "supabase",
        "tabla": table,
    }
    if total_en_bd is not None:
        result["total_en_bd"] = total_en_bd
    if total_monto_devuelto is not None:
        result["total_monto_devuelto"] = total_monto_devuelto
        if hay_mas:
            result["advertencia_total"] = "El total_monto_devuelto incluye solo las órdenes mostradas, no todas las del período."
    return result


def _stub_count_sales_orders_by_status(desde: str, hasta: str) -> dict[str, Any]:
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "conteo_por_estado": {"partial_delivery": 1, "delivered": 1},
        "total_ordenes": 2,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _count_sales_orders_by_status_from_supabase(desde: str, hasta: str) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _orders_list_table()
    date_c = _orders_list_date_column()
    status_c = _orders_list_status_column()

    page = 1000
    start = 0
    counts: dict[str, int] = {}
    total = 0

    while True:
        r = (
            client.table(table)
            .select(status_c)
            .gte(date_c, desde)
            .lte(date_c, hasta)
            .range(start, start + page - 1)
            .execute()
        )
        rows: list[dict[str, Any]] = r.data or []
        for row in rows:
            key = str(row.get(status_c) or "sin_estado")
            counts[key] = counts.get(key, 0) + 1
            total += 1
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "conteo_por_estado": ordered,
        "total_ordenes": total,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_top_selling_products(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "product_id": "demo-prod-1",
            "sku_o_codigo": "SKU-001",
            "nombre": "Producto demo A",
            "cantidad_vendida": 120.0,
            "monto_vendido": 450000.0,
            "costo_mercaderia_estimado": 315000.0,
            "utilidad_estimada": 135000.0,
            "markup_pct": 42.9,
            "pct_utilidad_ventas": 30.0,
        },
        {
            "product_id": "demo-prod-2",
            "sku_o_codigo": "SKU-002",
            "nombre": "Producto demo B",
            "cantidad_vendida": 95.0,
            "monto_vendido": 330000.0,
            "costo_mercaderia_estimado": 231000.0,
            "utilidad_estimada": 99000.0,
            "markup_pct": 42.9,
            "pct_utilidad_ventas": 30.0,
        },
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "top_productos": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota_costo": "Costo estimado basado en cost_price actual del producto (c/ IVA). Para margen exacto usar el Reporte de Rentabilidad del ERP.",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _top_selling_products_from_supabase(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    items_t, amount_c, fk_c, orders_t, date_c, oid_c = _sales_items_column_config()
    product_c, qty_c = _sales_items_top_columns()

    page = 1000
    start = 0
    order_ids: list[Any] = []
    while True:
        r = (
            client.table(orders_t)
            .select(oid_c)
            .gte(date_c, desde)
            .lte(date_c, hasta)
            .range(start, start + page - 1)
            .execute()
        )
        rows = r.data or []
        for row in rows:
            oid = row.get(oid_c)
            if oid is not None:
                order_ids.append(oid)
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    order_ids = list(dict.fromkeys(order_ids))
    if not order_ids:
        return {
            "periodo": {"desde": desde, "hasta": hasta},
            "top_productos": [],
            "cantidad_devuelta": 0,
            "limite": limit,
            "fuente": "supabase",
            "tabla_items": items_t,
        }

    try:
        chunk_sz = int(os.environ.get("ERP_SUPABASE_SALES_ITEMS_ORDER_ID_CHUNK", "80") or "80")
    except ValueError:
        chunk_sz = 80
    chunk_sz = max(1, min(chunk_sz, 200))

    agg: dict[Any, dict[str, float]] = {}
    for i in range(0, len(order_ids), chunk_sz):
        chunk = order_ids[i : i + chunk_sz]
        i_start = 0
        while True:
            r = (
                client.table(items_t)
                .select(f"{product_c},{qty_c},{amount_c}")
                .in_(fk_c, chunk)
                .range(i_start, i_start + page - 1)
                .execute()
            )
            rows = r.data or []
            for row in rows:
                pid = row.get(product_c)
                if pid is None:
                    continue
                cur = agg.setdefault(pid, {"cantidad_vendida": 0.0, "monto_vendido": 0.0})
                cur["cantidad_vendida"] += _to_float(row.get(qty_c))
                cur["monto_vendido"] += _to_float(row.get(amount_c))
            if len(rows) < page:
                break
            i_start += page
            if i_start > 500_000:
                break

    ids = list(agg.keys())
    code_c, name_c, _, uuid_c = _product_table_columns()
    by_id: dict[Any, dict[str, Any]] = {}

    # Columnas de costo del producto (fallback para estimación de margen)
    prod_cost_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_PRICE") or "cost_price").strip()
    prod_cost_tax_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_INCLUDES_TAX") or "cost_price_includes_tax").strip()
    prod_vat_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_VAT_RATE") or "vat_rate").strip()

    if uuid_c and ids:
        prod_select = f"{uuid_c},{code_c},{name_c}"
        try:
            # Intentar traer también columnas de costo
            prod_select_cost = f"{prod_select},{prod_cost_col},{prod_cost_tax_col},{prod_vat_col}"
            for i in range(0, len(ids), 200):
                chunk = ids[i : i + 200]
                r = client.table(_products_table()).select(prod_select_cost).in_(uuid_c, chunk).execute()
                for prow in r.data or []:
                    by_id[prow.get(uuid_c)] = prow
        except Exception:
            # Fallback sin columnas de costo
            for i in range(0, len(ids), 200):
                chunk = ids[i : i + 200]
                r = client.table(_products_table()).select(prod_select).in_(uuid_c, chunk).execute()
                for prow in r.data or []:
                    by_id[prow.get(uuid_c)] = prow

    ranked = sorted(
        agg.items(),
        key=lambda kv: (-kv[1]["cantidad_vendida"], -kv[1]["monto_vendido"]),
    )
    out: list[dict[str, Any]] = []
    for pid, vals in ranked[:limit]:
        p = by_id.get(pid, {})
        cantidad = vals["cantidad_vendida"]
        monto = vals["monto_vendido"]

        # Estimación de costo con IVA usando cost_price del producto
        base_cost = _to_float(p.get(prod_cost_col))
        includes_tax = bool(p.get(prod_cost_tax_col))
        vat_rate = _to_float(p.get(prod_vat_col))
        tax_mult = 1.0 if includes_tax else (1.0 + vat_rate / 100.0)
        costo_estimado = round(cantidad * base_cost * tax_mult, 2)
        utilidad_estimada = round(monto - costo_estimado, 2)
        markup_pct = round((utilidad_estimada / costo_estimado) * 100, 2) if costo_estimado > 0 else 0.0
        pct_utilidad = round((utilidad_estimada / monto) * 100, 2) if monto > 0 else 0.0

        out.append(
            {
                "product_id": pid,
                "sku_o_codigo": p.get(code_c),
                "nombre": p.get(name_c),
                "cantidad_vendida": round(cantidad, 3),
                "monto_vendido": round(monto, 2),
                "costo_mercaderia_estimado": costo_estimado,
                "utilidad_estimada": utilidad_estimada,
                "markup_pct": markup_pct,
                "pct_utilidad_ventas": pct_utilidad,
            }
        )

    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "top_productos": out,
        "cantidad_devuelta": len(out),
        "limite": limit,
        "fuente": "supabase",
        "tabla_items": items_t,
        "nota_costo": (
            "Costo estimado basado en el cost_price actual del producto (c/ IVA). "
            "Para un ranking por ganancia exacta, usar el Reporte de Rentabilidad del ERP."
        ),
    }


def _stub_least_selling_products(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "product_id": "demo-prod-3",
            "sku_o_codigo": "SKU-003",
            "nombre": "Producto demo C",
            "cantidad_vendida": 2.0,
            "monto_vendido": 1500.0,
        },
        {
            "product_id": "demo-prod-4",
            "sku_o_codigo": "SKU-004",
            "nombre": "Producto demo D",
            "cantidad_vendida": 5.0,
            "monto_vendido": 4000.0,
        },
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "productos_menos_vendidos": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _least_selling_products_from_supabase(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    items_t, amount_c, fk_c, orders_t, date_c, oid_c = _sales_items_column_config()
    product_c, qty_c = _sales_items_top_columns()

    page = 1000
    start = 0
    order_ids: list[Any] = []
    while True:
        r = (
            client.table(orders_t)
            .select(oid_c)
            .gte(date_c, desde)
            .lte(date_c, hasta)
            .range(start, start + page - 1)
            .execute()
        )
        rows = r.data or []
        for row in rows:
            oid = row.get(oid_c)
            if oid is not None:
                order_ids.append(oid)
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    order_ids = list(dict.fromkeys(order_ids))
    if not order_ids:
        return {
            "periodo": {"desde": desde, "hasta": hasta},
            "productos_menos_vendidos": [],
            "cantidad_devuelta": 0,
            "limite": limit,
            "fuente": "supabase",
            "tabla_items": items_t,
        }

    try:
        chunk_sz = int(os.environ.get("ERP_SUPABASE_SALES_ITEMS_ORDER_ID_CHUNK", "80") or "80")
    except ValueError:
        chunk_sz = 80
    chunk_sz = max(1, min(chunk_sz, 200))

    agg: dict[Any, dict[str, float]] = {}
    for i in range(0, len(order_ids), chunk_sz):
        chunk = order_ids[i : i + chunk_sz]
        i_start = 0
        while True:
            r = (
                client.table(items_t)
                .select(f"{product_c},{qty_c},{amount_c}")
                .in_(fk_c, chunk)
                .range(i_start, i_start + page - 1)
                .execute()
            )
            rows = r.data or []
            for row in rows:
                pid = row.get(product_c)
                if pid is None:
                    continue
                cur = agg.setdefault(pid, {"cantidad_vendida": 0.0, "monto_vendido": 0.0})
                cur["cantidad_vendida"] += _to_float(row.get(qty_c))
                cur["monto_vendido"] += _to_float(row.get(amount_c))
            if len(rows) < page:
                break
            i_start += page
            if i_start > 500_000:
                break

    ids = list(agg.keys())
    code_c, name_c, _, uuid_c = _product_table_columns()
    by_id: dict[Any, dict[str, Any]] = {}
    if uuid_c and ids:
        for i in range(0, len(ids), 200):
            chunk = ids[i : i + 200]
            r = client.table(_products_table()).select(f"{uuid_c},{code_c},{name_c}").in_(uuid_c, chunk).execute()
            for prow in r.data or []:
                by_id[prow.get(uuid_c)] = prow

    ranked = sorted(
        agg.items(),
        key=lambda kv: (kv[1]["cantidad_vendida"], kv[1]["monto_vendido"]),
    )
    out: list[dict[str, Any]] = []
    for pid, vals in ranked[:limit]:
        p = by_id.get(pid, {})
        out.append(
            {
                "product_id": pid,
                "sku_o_codigo": p.get(code_c),
                "nombre": p.get(name_c),
                "cantidad_vendida": round(vals["cantidad_vendida"], 3),
                "monto_vendido": round(vals["monto_vendido"], 2),
            }
        )

    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "productos_menos_vendidos": out,
        "cantidad_devuelta": len(out),
        "limite": limit,
        "fuente": "supabase",
        "tabla_items": items_t,
    }


def _stub_purchase_summary(desde: str, hasta: str) -> dict[str, Any]:
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_compras": 49610.0,
        "cantidad_ordenes": 1,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _purchase_summary_from_supabase(desde: str, hasta: str) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _purchase_orders_table()
    date_c = _po_date_column()
    amount_c = _po_amount_column()
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)
    r = client.table(table).select(amount_c).gte(date_c, desde_filtro).lte(date_c, hasta_filtro).execute()
    rows: list[dict[str, Any]] = r.data or []
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(amount_c) or 0)
        except (TypeError, ValueError):
            continue
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_compras": total,
        "cantidad_ordenes": len(rows),
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_list_purchase_orders(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "30c95614-388d-4e7d-b975-558141d0834a",
            "number": "OC-2026-611563",
            "supplier_id": "99e8c129-99b0-4ef4-892c-4d1d20396d13",
            "order_date": desde,
            "expected_date": hasta,
            "total_amount": 49610.0,
            "status": "pending",
            "order_type": "direct",
        }
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "ordenes_compra": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _list_purchase_orders_from_supabase(desde: str, hasta: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _purchase_orders_table()
    date_c = _po_date_column()
    select_cols = _po_list_select_expr()
    supplier_col = _po_supplier_id_column()
    select_parts = [p.strip() for p in select_cols.split(",") if p.strip()]
    if _po_include_supplier_data() and supplier_col not in select_parts:
        select_parts.append(supplier_col)
    select_cols = ",".join(select_parts)
    lim = max(1, min(limit, 200))
    r = (
        client.table(table)
        .select(select_cols)
        .gte(date_c, desde)
        .lte(date_c, hasta)
        .order(date_c, desc=True)
        .limit(lim)
        .execute()
    )
    rows: list[dict[str, Any]] = r.data or []
    if _po_include_supplier_data():
        rows = _attach_suppliers_to_purchase_orders(rows, supplier_col)
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "ordenes_compra": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_count_purchase_orders_by_status(desde: str, hasta: str) -> dict[str, Any]:
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "conteo_por_estado": {"pending": 1},
        "total_ordenes": 1,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _count_purchase_orders_by_status_from_supabase(desde: str, hasta: str) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _purchase_orders_table()
    date_c = _po_date_column()
    status_c = _po_status_column()

    page = 1000
    start = 0
    counts: dict[str, int] = {}
    total = 0

    while True:
        r = (
            client.table(table)
            .select(status_c)
            .gte(date_c, desde)
            .lte(date_c, hasta)
            .range(start, start + page - 1)
            .execute()
        )
        rows: list[dict[str, Any]] = r.data or []
        for row in rows:
            key = str(row.get(status_c) or "sin_estado")
            counts[key] = counts.get(key, 0) + 1
            total += 1
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "conteo_por_estado": ordered,
        "total_ordenes": total,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_list_purchase_order_items(purchase_order_id: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "id": "0e560486-e1e6-4562-8784-457ce70f8fc9",
            "purchase_order_id": purchase_order_id.strip(),
            "product_id": "632f6c22-b14f-4950-b7b5-9af81f438213",
            "quantity": 15.0,
            "unit_price": 49176.9753,
            "total_price": 737654.63,
            "received_quantity": 15.0,
            "unit": "caja",
        }
    ]
    return {
        "purchase_order_id": purchase_order_id.strip(),
        "lineas": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _list_purchase_order_items_from_supabase(purchase_order_id: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _purchase_order_items_table()
    fk_c = _po_items_fk_column()
    select_cols = _po_items_select_expr()
    lim = max(1, min(limit, 500))
    r = (
        client.table(table)
        .select(select_cols)
        .eq(fk_c, purchase_order_id.strip())
        .limit(lim)
        .execute()
    )
    rows: list[dict[str, Any]] = r.data or []
    return {
        "purchase_order_id": purchase_order_id.strip(),
        "lineas": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_customers(query: str, limit: int) -> dict[str, Any]:
    q = query.lower().strip()
    catalog = [
        {
            "id": "demo-cus-1",
            "razon_social": "Cliente demo S.R.L.",
            "tax_id": "20-30000000-1",
            "contacto": "Juan Demo",
            "codigo": "C-100",
            "email": "ventas@cliente-demo.com",
            "telefono": "+54 11 1111-1111",
            "activo": True,
        },
    ]
    hits = []
    for c in catalog:
        blob = " ".join(str(c.get(k) or "") for k in c).lower()
        if q in blob:
            hits.append(c)
    return {
        "query": query,
        "resultados": hits[:limit],
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _attach_customers_to_orders(rows: list[dict[str, Any]], customer_id_col: str) -> list[dict[str, Any]]:
    customer_ids = [row.get(customer_id_col) for row in rows if row.get(customer_id_col) is not None]
    if not customer_ids:
        return rows

    customer_ids = list(dict.fromkeys(customer_ids))
    client = _get_supabase()
    assert client is not None
    table, id_c, name_c, tax_c, _, _, _, _, _ = _customer_column_config()

    by_id: dict[Any, dict[str, Any]] = {}
    chunk_sz = 200
    for i in range(0, len(customer_ids), chunk_sz):
        chunk = customer_ids[i : i + chunk_sz]
        r = client.table(table).select(f"{id_c},{name_c},{tax_c}").in_(id_c, chunk).execute()
        for c in r.data or []:
            by_id[c.get(id_c)] = c

    out: list[dict[str, Any]] = []
    for row in rows:
        rid = row.get(customer_id_col)
        c = by_id.get(rid)
        if c:
            row = dict(row)
            row["customer_business_name"] = c.get(name_c)
            row["customer_tax_id"] = c.get(tax_c)
        out.append(row)
    return out


def _attach_suppliers_to_purchase_orders(
    rows: list[dict[str, Any]], supplier_id_col: str
) -> list[dict[str, Any]]:
    supplier_ids = [row.get(supplier_id_col) for row in rows if row.get(supplier_id_col) is not None]
    if not supplier_ids:
        return rows

    supplier_ids = list(dict.fromkeys(supplier_ids))
    client = _get_supabase()
    assert client is not None
    table, id_c, name_c, tax_c, _, _, _, _ = _supplier_column_config()

    by_id: dict[Any, dict[str, Any]] = {}
    chunk_sz = 200
    for i in range(0, len(supplier_ids), chunk_sz):
        chunk = supplier_ids[i : i + chunk_sz]
        r = client.table(table).select(f"{id_c},{name_c},{tax_c}").in_(id_c, chunk).execute()
        for srow in r.data or []:
            by_id[srow.get(id_c)] = srow

    out: list[dict[str, Any]] = []
    for row in rows:
        sid = row.get(supplier_id_col)
        s = by_id.get(sid)
        if s:
            row = dict(row)
            row["supplier_business_name"] = s.get(name_c)
            row["supplier_tax_id"] = s.get(tax_c)
        out.append(row)
    return out


def _stub_sales(desde: str, hasta: str) -> dict[str, Any]:
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_ventas": 125430.50,
        "cantidad_documentos": 42,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _stub_suppliers(query: str, limit: int) -> dict[str, Any]:
    q = query.lower().strip()
    catalog = [
        {
            "id": "demo-sup-1",
            "razon_social": "Proveedor demo S.A.",
            "tax_id": "30-70000000-9",
            "contacto": "María Pérez",
            "email": "compras@demo.com",
            "telefono": "+54 11 0000-0000",
            "activo": True,
        },
    ]
    hits = []
    for p in catalog:
        blob = " ".join(str(p.get(k) or "") for k in p).lower()
        if q in blob:
            hits.append(p)
    return {
        "query": query,
        "resultados": hits[:limit],
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _stub_products(query: str, limit: int) -> dict[str, Any]:
    q = query.lower().strip()
    catalog = [
        {"id": "SKU-001", "nombre": "Producto demo A", "precio": 100.0},
        {"id": "SKU-002", "nombre": "Producto demo B", "precio": 250.5},
    ]
    hits = [p for p in catalog if q in p["nombre"].lower() or q in p["id"].lower()]
    return {
        "query": query,
        "resultados": hits[:limit],
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _sales_from_supabase_orders(client: Any, desde: str, hasta: str) -> dict[str, Any]:
    table, date_c, amount_c = _sales_orders_column_config()
    q = client.table(table).select(amount_c)
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)
    r = q.gte(date_c, desde_filtro).lte(date_c, hasta_filtro).execute()
    rows: list[dict[str, Any]] = r.data or []
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(amount_c) or 0)
        except (TypeError, ValueError):
            continue
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_ventas": total,
        "cantidad_documentos": len(rows),
        "fuente": "supabase",
        "modo": "orders",
        "tabla": table,
    }


def _sales_from_supabase_items(client: Any, desde: str, hasta: str) -> dict[str, Any]:
    """Suma líneas: primero pedidos en rango de fechas, luego ítems con FK en esos ids (sin embed)."""
    items_t, amount_c, fk_c, orders_t, date_c, oid_c = _sales_items_column_config()
    page = 1000
    start = 0
    order_ids: list[Any] = []
    while True:
        desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)
        oq = client.table(orders_t).select(oid_c).gte(date_c, desde_filtro).lte(date_c, hasta_filtro)
        orow = oq.range(start, start + page - 1).execute()
        batch = orow.data or []
        for row in batch:
            oid = row.get(oid_c)
            if oid is not None:
                order_ids.append(oid)
        if len(batch) < page:
            break
        start += page
        if start > 500_000:
            break

    order_set = set(order_ids)
    total = 0.0
    line_count = 0
    try:
        chunk_sz = int(os.environ.get("ERP_SUPABASE_SALES_ITEMS_ORDER_ID_CHUNK", "80") or "80")
    except ValueError:
        chunk_sz = 80
    chunk_sz = max(1, min(chunk_sz, 200))

    for i in range(0, len(order_ids), chunk_sz):
        chunk = order_ids[i : i + chunk_sz]
        i_start = 0
        while True:
            iq = client.table(items_t).select(amount_c).in_(fk_c, chunk)
            ir = iq.range(i_start, i_start + page - 1).execute()
            ibatch = ir.data or []
            for row in ibatch:
                line_count += 1
                try:
                    total += float(row.get(amount_c) or 0)
                except (TypeError, ValueError):
                    continue
            if len(ibatch) < page:
                break
            i_start += page
            if i_start > 500_000:
                break

    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "total_ventas": total,
        "cantidad_documentos": len(order_set),
        "cantidad_lineas": line_count,
        "fuente": "supabase",
        "modo": "items",
        "tabla": items_t,
        "tabla_pedidos": orders_t,
    }


def _sales_from_supabase(desde: str, hasta: str) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    if _sales_source_mode() == "items":
        return _sales_from_supabase_items(client, desde, hasta)
    return _sales_from_supabase_orders(client, desde, hasta)


def _suppliers_from_supabase(query: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table, id_c, name_c, tax_c, email_c, phone_c, active_c, contact_c = _supplier_column_config()
    extra_cols = _supplier_extra_col_names()
    select_cols = _supplier_select_columns(
        id_c, name_c, tax_c, email_c, phone_c, active_c, contact_c, extra_cols
    )
    tok = _like_token(query)
    pat = f"%{tok}%"
    parts = [f"{name_c}.ilike.{pat}", f"{tax_c}.ilike.{pat}"]
    if contact_c:
        parts.append(f"{contact_c}.ilike.{pat}")
    or_clause = ",".join(parts)
    q = client.table(table).select(select_cols)
    if _env_flag("ERP_SUPABASE_SUPPLIERS_ONLY_ACTIVE"):
        q = q.eq(active_c, True)
    r = q.or_(or_clause).limit(limit).execute()
    raw_rows: list[dict[str, Any]] = r.data or []
    rows = [
        _normalize_supplier_row(
            row, id_c, name_c, tax_c, email_c, phone_c, active_c, contact_c, extra_cols
        )
        for row in raw_rows
    ]
    return {
        "query": query,
        "resultados": rows,
        "fuente": "supabase",
        "tabla": table,
    }


def _customers_from_supabase(query: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table, id_c, name_c, tax_c, email_c, phone_c, active_c, code_c, contact_c = _customer_column_config()
    extra_cols = _customer_extra_col_names()
    select_cols = _customer_select_columns(
        id_c, name_c, tax_c, email_c, phone_c, active_c, code_c, contact_c, extra_cols
    )
    tok = _like_token(query)
    pat = f"%{tok}%"
    parts = [f"{name_c}.ilike.{pat}", f"{tax_c}.ilike.{pat}", f"{code_c}.ilike.{pat}"]
    if contact_c:
        parts.append(f"{contact_c}.ilike.{pat}")
    or_clause = ",".join(parts)
    q = client.table(table).select(select_cols)
    if _env_flag("ERP_SUPABASE_CUSTOMERS_ONLY_ACTIVE"):
        q = q.eq(active_c, True)
    r = q.or_(or_clause).limit(limit).execute()
    raw_rows: list[dict[str, Any]] = r.data or []
    rows = [
        _normalize_customer_row(
            row, id_c, name_c, tax_c, email_c, phone_c, active_c, code_c, contact_c, extra_cols
        )
        for row in raw_rows
    ]
    return {
        "query": query,
        "resultados": rows,
        "fuente": "supabase",
        "tabla": table,
    }


def _products_from_supabase(query: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _products_table()
    code_c, name_c, price_c, uuid_c = _product_table_columns()
    select_cols = _product_select_list(code_c, name_c, price_c, uuid_c)
    tok = _like_token(query)
    pat = f"%{tok}%"
    or_clause = f"{name_c}.ilike.{pat},{code_c}.ilike.{pat}"
    print(f"[DEBUG] SELECT {select_cols} FROM {table} WHERE {or_clause} LIMIT {limit}", flush=True)
    q = client.table(table).select(select_cols)
    r = q.or_(or_clause).limit(limit).execute()
    raw_rows: list[dict[str, Any]] = r.data or []
    rows = [_normalize_product_row(row, code_c, name_c, price_c, uuid_c) for row in raw_rows]
    return {
        "query": query,
        "resultados": rows,
    }


def _stub_product_available_stock(query: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "product_id": "64a92a94-f913-4f78-b95c-4ef1b51030ce",
            "sku_o_codigo": query[:16] or "DEMO-SKU",
            "nombre": "Producto demo",
            "stock_total": 15.0,
            "stock_reservado": 0.0,
            "stock_disponible": 15.0,
            "min_stock_total": 0.0,
            "stock_projected_total": 10.0,
            "depositos": [
                {
                    "warehouse_id": "be491813-a3f0-4374-8bb3-21ca2d49d7d4",
                    "stock": 15.0,
                    "reserved_stock": 0.0,
                    "available_stock": 15.0,
                    "min_stock": 0.0,
                    "stock_projected": 10.0,
                }
            ],
        }
    ]
    return {
        "query": query,
        "resultados": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _product_available_stock_from_supabase(query: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    product_rows = _resolve_products_by_query(query, max(1, min(limit, 20)))
    if not product_rows:
        return {
            "query": query,
            "resultados": [],
            "nota": "sin coincidencias de producto",
        }
    code_c, name_c, _, uuid_c = _product_table_columns()
    assert uuid_c is not None
    product_map: dict[Any, dict[str, Any]] = {row.get(uuid_c): row for row in product_rows if row.get(uuid_c) is not None}
    product_ids = list(product_map.keys())

    ws_table, ws_product_c, ws_warehouse_c, ws_stock_c, ws_reserved_c, ws_min_c, ws_projected_c = (
        _warehouse_stock_column_config()
    )
    select_cols = ",".join((ws_product_c, ws_warehouse_c, ws_stock_c, ws_reserved_c, ws_min_c, ws_projected_c))
    r = client.table(ws_table).select(select_cols).in_(ws_product_c, product_ids).execute()
    stock_rows: list[dict[str, Any]] = r.data or []

    by_product: dict[Any, list[dict[str, Any]]] = {}
    for row in stock_rows:
        pid = row.get(ws_product_c)
        if pid is None:
            continue
        by_product.setdefault(pid, []).append(row)

    out: list[dict[str, Any]] = []
    for pid in product_ids:
        product = product_map.get(pid) or {}
        rows = by_product.get(pid, [])
        stock_total = 0.0
        reserved_total = 0.0
        min_total = 0.0
        projected_total = 0.0
        depots: list[dict[str, Any]] = []
        for row in rows:
            stock_v = _to_float(row.get(ws_stock_c))
            reserved_v = _to_float(row.get(ws_reserved_c))
            min_v = _to_float(row.get(ws_min_c))
            projected_v = _to_float(row.get(ws_projected_c))
            stock_total += stock_v
            reserved_total += reserved_v
            min_total += min_v
            projected_total += projected_v
            depots.append(
                {
                    "warehouse_id": row.get(ws_warehouse_c),
                    "stock": stock_v,
                    "reserved_stock": reserved_v,
                    "available_stock": stock_v - reserved_v,
                    "min_stock": min_v,
                    "stock_projected": projected_v,
                }
            )

        out.append(
            {
                "sku": product.get(code_c),
                "nombre": product.get(name_c),
                "stock_total": stock_total,
                "stock_reservado": reserved_total,
                "stock_disponible": stock_total - reserved_total,
                "min_stock": min_total,
                "stock_proyectado": projected_total,
                "depositos": [{k: v for k, v in d.items() if k != "warehouse_id" or len(depots) > 1} for d in depots],
            }
        )
    return {
        "query": query,
        "resultados": out,
    }


def _stub_products_below_min_stock(limit: int) -> dict[str, Any]:
    demo = [
        {
            "product_id": "64a92a94-f913-4f78-b95c-4ef1b51030ce",
            "warehouse_id": "be491813-a3f0-4374-8bb3-21ca2d49d7d4",
            "sku_o_codigo": "DEMO-001",
            "nombre": "Producto demo",
            "stock": 2.0,
            "reserved_stock": 0.0,
            "stock_disponible": 2.0,
            "min_stock": 5.0,
            "falta_para_minimo": 3.0,
        }
    ]
    return {
        "resultados": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _products_below_min_stock_from_supabase(limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    _, code_c, name_c, uuid_c = _product_table_columns()
    if not uuid_c:
        raise ValueError("ERP_SUPABASE_PRODUCT_COL_UUID es obligatorio para consultas de stock mínimo")
    ws_table, ws_product_c, ws_warehouse_c, ws_stock_c, ws_reserved_c, ws_min_c, _ = _warehouse_stock_column_config()
    select_cols = ",".join((ws_product_c, ws_warehouse_c, ws_stock_c, ws_reserved_c, ws_min_c))
    r = client.table(ws_table).select(select_cols).limit(max(1, min(limit * 8, 1000))).execute()
    rows: list[dict[str, Any]] = r.data or []

    filtered: list[dict[str, Any]] = []
    for row in rows:
        stock_v = _to_float(row.get(ws_stock_c))
        reserved_v = _to_float(row.get(ws_reserved_c))
        min_v = _to_float(row.get(ws_min_c))
        available_v = stock_v - reserved_v
        if available_v <= min_v:
            rr = dict(row)
            rr["_available_stock"] = available_v
            filtered.append(rr)

    pids = [row.get(ws_product_c) for row in filtered if row.get(ws_product_c) is not None]
    pids = list(dict.fromkeys(pids))
    by_id: dict[Any, dict[str, Any]] = {}
    if pids:
        pr = client.table(_products_table()).select(f"{uuid_c},{code_c},{name_c}").in_(uuid_c, pids).execute()
        for prow in pr.data or []:
            by_id[prow.get(uuid_c)] = prow

    out: list[dict[str, Any]] = []
    for row in filtered[:limit]:
        pid = row.get(ws_product_c)
        prod = by_id.get(pid, {})
        min_v = _to_float(row.get(ws_min_c))
        available_v = _to_float(row.get("_available_stock"))
        out.append(
            {
                "product_id": pid,
                "warehouse_id": row.get(ws_warehouse_c),
                "sku_o_codigo": prod.get(code_c),
                "nombre": prod.get(name_c),
                "stock": _to_float(row.get(ws_stock_c)),
                "reserved_stock": _to_float(row.get(ws_reserved_c)),
                "stock_disponible": available_v,
                "min_stock": min_v,
                "falta_para_minimo": max(0.0, min_v - available_v),
            }
        )

    return {
        "resultados": out,
        "cantidad_devuelta": len(out),
        "fuente": "supabase",
        "tabla_stock": ws_table,
    }


def _stub_top_products_by_stock(limit: int, order: str) -> dict[str, Any]:
    demo = [
        {"sku": "SKU-DEMO-001", "nombre": "Producto Demo A", "stock_disponible": 500.0, "stock_reservado": 10.0},
        {"sku": "SKU-DEMO-002", "nombre": "Producto Demo B", "stock_disponible": 320.0, "stock_reservado": 0.0},
    ]
    return {
        "productos": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "orden": order,
        "fuente": "stub",
    }


def _stub_products_without_stock_days(days: int, limit: int) -> dict[str, Any]:
    return {
        "resultados": [
            {"sku_o_codigo": "DEMO-001", "nombre": "Producto Demo", "stock": 0, "updated_at": "2024-01-01", "dias_sin_stock": days}
        ],
        "cantidad_devuelta": 1,
        "dias_sin_stock": days,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer datos reales",
    }


def _products_without_stock_days_from_supabase(days: int, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    ws_table, ws_product_c, ws_warehouse_c, ws_stock_c, _, _, _ = _warehouse_stock_column_config()
    updated_at_c = (
        os.environ.get("ERP_SUPABASE_WAREHOUSE_STOCK_UPDATED_AT_COL", "updated_at") or "updated_at"
    ).strip()
    _, code_c, name_c, uuid_c = _product_table_columns()

    cutoff = (date.today() - timedelta(days=days)).isoformat()

    select_cols = ",".join(filter(None, [ws_product_c, ws_warehouse_c, ws_stock_c, updated_at_c]))
    r = (
        client.table(ws_table)
        .select(select_cols)
        .eq(ws_stock_c, 0)
        .lte(updated_at_c, cutoff)
        .limit(max(1, min(limit * 2, 500)))
        .execute()
    )
    rows: list[dict[str, Any]] = r.data or []

    pids = list(dict.fromkeys(row.get(ws_product_c) for row in rows if row.get(ws_product_c)))
    by_id: dict[Any, dict[str, Any]] = {}
    if pids and uuid_c:
        chunk_sz = 200
        for i in range(0, len(pids), chunk_sz):
            pr = client.table(_products_table()).select(f"{uuid_c},{code_c},{name_c}").in_(uuid_c, pids[i:i+chunk_sz]).execute()
            for p in pr.data or []:
                by_id[p.get(uuid_c)] = p

    out: list[dict[str, Any]] = []
    seen_pids: set = set()
    for row in rows:
        pid = row.get(ws_product_c)
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        prod = by_id.get(pid, {})
        out.append({
            "product_id": pid,
            "sku_o_codigo": prod.get(code_c),
            "nombre": prod.get(name_c),
            "stock": _to_float(row.get(ws_stock_c)),
            "updated_at": row.get(updated_at_c),
        })
        if len(out) >= limit:
            break

    return {
        "resultados": out,
        "cantidad_devuelta": len(out),
        "dias_sin_stock": days,
        "corte_fecha": cutoff,
        "fuente": "supabase",
    }


def _top_products_by_stock_from_supabase(
    limit: int, order: str, warehouse_name: str | None = None, only_principal: bool = False
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    ws_table, ws_product_c, ws_warehouse_c, ws_stock_c, ws_reserved_c, ws_min_c, _ = _warehouse_stock_column_config()
    _, code_c, name_c, uuid_c = _product_table_columns()
    if not uuid_c:
        raise ValueError("ERP_SUPABASE_PRODUCT_COL_UUID es obligatorio para consultas de stock")

    filter_wh_id: str | None = None
    if warehouse_name or only_principal:
        wh_query = client.table("warehouses").select("id,name,type").eq("is_active", True)
        if only_principal:
            wh_query = wh_query.eq("type", "principal")
        elif warehouse_name:
            wh_query = wh_query.filter("name", "ilike", f"%{warehouse_name.strip()}%")
        wh_r = wh_query.limit(1).execute()
        wh_rows = wh_r.data or []
        if not wh_rows:
            label = "principal" if only_principal else warehouse_name
            raise ValueError(f"No se encontró almacén: {label!r}")
        filter_wh_id = wh_rows[0]["id"]
        warehouse_label = wh_rows[0]["name"]
    else:
        warehouse_label = "todos"

    lim = max(1, min(limit, 100))
    desc = order.lower() != "asc"

    q = client.table(ws_table).select(f"{ws_product_c},{ws_stock_c},{ws_reserved_c}")
    if filter_wh_id:
        q = q.eq(ws_warehouse_c, filter_wh_id)
    q = q.gt(ws_stock_c, 0).order(ws_stock_c, desc=desc).limit(lim)
    stock_rows: list[dict[str, Any]] = q.execute().data or []

    product_ids = [r[ws_product_c] for r in stock_rows if r.get(ws_product_c)]
    prod_map: dict[str, dict[str, Any]] = {}
    if product_ids:
        chunk_sz = 100
        for i in range(0, len(product_ids), chunk_sz):
            chunk = product_ids[i : i + chunk_sz]
            p_r = client.table("products").select(f"{uuid_c},{code_c},{name_c}").in_(uuid_c, chunk).execute()
            for p in p_r.data or []:
                prod_map[p[uuid_c]] = p

    results: list[dict[str, Any]] = []
    for row in stock_rows:
        pid = row.get(ws_product_c)
        prod = prod_map.get(pid) or {}
        stock_v = _to_float(row.get(ws_stock_c))
        reserved_v = _to_float(row.get(ws_reserved_c))
        results.append({
            "sku": prod.get(code_c),
            "nombre": prod.get(name_c),
            "stock_disponible": round(stock_v - reserved_v, 3),
            "stock_total": round(stock_v, 3),
            "stock_reservado": round(reserved_v, 3),
        })

    return {
        "productos": results,
        "cantidad_devuelta": len(results),
        "orden": "mayor_primero" if desc else "menor_primero",
        "almacen": warehouse_label,
        "fuente": "supabase",
    }


def _stub_recent_product_movements(query: str, days: int, limit: int) -> dict[str, Any]:
    now = datetime.utcnow().isoformat()
    demo = [
        {
            "fecha_hora": now,
            "tipo_movimiento": "Ajuste",
            "cantidad_entrada": 0.0,
            "cantidad_salida": 0.0,
            "sku": query[:16] or "DEMO-001",
            "descripcion_producto": "Movimiento demo",
        }
    ]
    return {
        "query": query,
        "dias": days,
        "movimientos": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _recent_product_movements_from_supabase(query: str, days: int, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    product_rows = _resolve_products_by_query(query, 10)
    _, code_c, name_c, uuid_c = _product_table_columns()
    if not uuid_c:
        raise ValueError("ERP_SUPABASE_PRODUCT_COL_UUID es obligatorio para consultas de cardex")
    product_map: dict[Any, dict[str, Any]] = {row.get(uuid_c): row for row in product_rows if row.get(uuid_c) is not None}
    pids = list(product_map.keys())
    if not pids:
        return {
            "query": query,
            "dias": days,
            "movimientos": [],
            "nota": "sin coincidencias de producto",
        }

    table, date_c, product_c, movement_c, in_c, out_c, sku_c, desc_c = _cardex_column_config()
    from_ts = (datetime.utcnow() - timedelta(days=days)).isoformat()
    select_cols = ",".join((date_c, product_c, movement_c, in_c, out_c, sku_c, desc_c))
    r = (
        client.table(table)
        .select(select_cols)
        .in_(product_c, pids)
        .gte(date_c, from_ts)
        .order(date_c, desc=True)
        .limit(limit)
        .execute()
    )
    rows: list[dict[str, Any]] = r.data or []
    out: list[dict[str, Any]] = []
    for row in rows:
        pid = row.get(product_c)
        p = product_map.get(pid, {})
        sku_prod = p.get(code_c)
        nombre_prod = p.get(name_c)
        sku_cx = row.get(sku_c)
        desc_cx = row.get(desc_c)
        entry: dict[str, Any] = {
            "sku": sku_prod or sku_cx,
            "nombre": nombre_prod or desc_cx,
            "fecha": row.get(date_c),
            "tipo": row.get(movement_c),
            "entrada": _to_float(row.get(in_c)),
            "salida": _to_float(row.get(out_c)),
        }
        if sku_cx and sku_cx != sku_prod:
            entry["sku_cardex"] = sku_cx
        if desc_cx and desc_cx != nombre_prod:
            entry["desc_cardex"] = desc_cx
        out.append(entry)
    return {
        "query": query,
        "dias": days,
        "movimientos": out,
    }


def _stub_top_sellers_by_invoicing(desde: str, hasta: str, metric: str, limit: int) -> dict[str, Any]:
    demo = [
        {
            "user_id": "demo-user-1", "nombre": "Ana Gómez",
            "total_facturado": 720000.0, "costo_mercaderia": 504000.0,
            "utilidad": 216000.0, "markup_pct": 42.9, "pct_utilidad_ventas": 30.0,
            "cantidad_facturas": 10,
        },
        {
            "user_id": "demo-user-2", "nombre": "Carlos Pérez",
            "total_facturado": 580000.0, "costo_mercaderia": 406000.0,
            "utilidad": 174000.0, "markup_pct": 42.9, "pct_utilidad_ventas": 30.0,
            "cantidad_facturas": 8,
        },
        {
            "user_id": "demo-user-3", "nombre": "María López",
            "total_facturado": 390000.0, "costo_mercaderia": 273000.0,
            "utilidad": 117000.0, "markup_pct": 42.9, "pct_utilidad_ventas": 30.0,
            "cantidad_facturas": 6,
        },
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "metrica": metric,
        "top_vendedores": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota_costo": "Costo de mercadería c/ IVA. Markup = Utilidad / Costo × 100.",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _top_sellers_by_invoicing_from_supabase(desde: str, hasta: str, metric: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    date_c = _invoices_date_column()
    amount_c = _invoices_amount_col()
    orders_table = _orders_list_table()
    seller_col = _orders_seller_col()

    items_t = _invoice_items_table()
    fk_c = _invoice_items_fk_col()
    product_id_c = _invoice_items_product_id_col()
    qty_c = _invoice_items_qty_col()
    purchase_cost_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_COST_COL") or "purchase_cost").strip()
    purchase_cost_tax_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_COST_INCLUDES_TAX_COL") or "purchase_cost_includes_tax").strip()
    purchase_vat_c = (os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_PURCHASE_VAT_RATE_COL") or "purchase_vat_rate").strip()

    # Paso 1: IDs que son destino de conversión
    page = 1000
    excluded_ids: set[str] = set()
    try:
        excl_start = 0
        while True:
            excl_r = (
                client.table(table)
                .select("converted_to_invoice_id")
                .not_.is_("converted_to_invoice_id", "null")
                .order("id")
                .range(excl_start, excl_start + page - 1)
                .execute()
            )
            excl_rows = excl_r.data or []
            for row in excl_rows:
                if row.get("converted_to_invoice_id"):
                    excluded_ids.add(row["converted_to_invoice_id"])
            if len(excl_rows) < page:
                break
            excl_start += page
            if excl_start > 500_000:
                break
    except Exception:
        pass

    # Paso 2: facturas válidas del período
    excluded_statuses = ["cancelled", "voided", "converted", "draft"]
    excluded_types = {"nota_pedido", "np"}
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)

    start = 0
    inv_rows: list[dict[str, Any]] = []
    while True:
        r = (
            client.table(table)
            .select(f"id,{amount_c},created_by,sales_order_id,warehouse_id,payment_condition,invoice_type")
            .gte(date_c, desde_filtro)
            .lte(date_c, hasta_filtro)
            .not_.in_("status", excluded_statuses)
            .order("id")
            .range(start, start + page - 1)
            .execute()
        )
        rows: list[dict[str, Any]] = r.data or []
        for row in rows:
            if row.get("id") in excluded_ids:
                continue
            if (row.get("invoice_type") or "").lower() in excluded_types:
                continue
            if not (row.get("warehouse_id") or row.get("sales_order_id") or row.get("payment_condition")):
                continue
            inv_rows.append(row)
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    # Paso 3: resolver so.created_by para facturas con sales_order_id (COALESCE)
    so_seller: dict[str, str] = {}
    so_ids = list({row["sales_order_id"] for row in inv_rows if row.get("sales_order_id")})
    if so_ids:
        for i in range(0, len(so_ids), 200):
            chunk = so_ids[i : i + 200]
            r = client.table(orders_table).select(f"id,{seller_col}").in_("id", chunk).execute()
            for srow in r.data or []:
                if srow.get(seller_col):
                    so_seller[srow["id"]] = srow[seller_col]

    # Paso 4: agregar por vendedor y construir mapa factura→vendedor
    invoice_to_seller: dict[str, Any] = {}
    agg: dict[Any, dict[str, Any]] = {}
    for row in inv_rows:
        so_id = row.get("sales_order_id")
        sid = (so_id and so_seller.get(so_id)) or row.get("created_by")
        if sid is None:
            continue
        cur = agg.setdefault(sid, {"total_facturado": 0.0, "cantidad_facturas": 0, "costo_total": 0.0})
        cur["total_facturado"] += _to_float(row.get(amount_c))
        cur["cantidad_facturas"] += 1
        if row.get("id"):
            invoice_to_seller[row["id"]] = sid

    # Paso 4.5: costos de ítems en chunks (por factura_id)
    all_inv_ids = list(invoice_to_seller.keys())
    if all_inv_ids:
        try:
            chunk_sz = int(os.environ.get("ERP_SUPABASE_INVOICE_ITEMS_INVOICE_ID_CHUNK", "80") or "80")
        except ValueError:
            chunk_sz = 80
        chunk_sz = max(1, min(chunk_sz, 200))

        item_select = f"{fk_c},{product_id_c},{qty_c},{purchase_cost_c},{purchase_cost_tax_c},{purchase_vat_c}"
        raw_items: list[dict[str, Any]] = []
        product_ids_needed: set[Any] = set()

        for i in range(0, len(all_inv_ids), chunk_sz):
            chunk = all_inv_ids[i : i + chunk_sz]
            i_start = 0
            while True:
                try:
                    r_i = client.table(items_t).select(item_select).in_(fk_c, chunk).range(i_start, i_start + 999).execute()
                except Exception:
                    r_i = client.table(items_t).select(f"{fk_c},{product_id_c},{qty_c},{purchase_cost_c}").in_(fk_c, chunk).range(i_start, i_start + 999).execute()
                rows_i: list[dict[str, Any]] = r_i.data or []
                for item in rows_i:
                    raw_items.append(item)
                    pid = item.get(product_id_c)
                    if pid and (item.get(purchase_cost_c) is None or item.get(purchase_cost_c) == ""):
                        product_ids_needed.add(pid)
                if len(rows_i) < 1000:
                    break
                i_start += 1000
                if i_start > 500_000:
                    break

        # Fallback cost_price del producto cuando el ítem no tiene purchase_cost
        products_cost: dict[Any, dict[str, Any]] = {}
        if product_ids_needed:
            _, _, _, uuid_c = _product_table_columns()
            if not uuid_c:
                uuid_c = "id"
            prod_cost_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_PRICE") or "cost_price").strip()
            prod_cost_tax_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_COST_INCLUDES_TAX") or "cost_price_includes_tax").strip()
            prod_vat_col = (os.environ.get("ERP_SUPABASE_PRODUCT_COL_VAT_RATE") or "vat_rate").strip()
            try:
                prod_r = (
                    client.table(_products_table())
                    .select(f"{uuid_c},{prod_cost_col},{prod_cost_tax_col},{prod_vat_col}")
                    .in_(uuid_c, list(product_ids_needed))
                    .execute()
                )
                for p in (prod_r.data or []):
                    p_id = p.get(uuid_c)
                    if p_id:
                        products_cost[p_id] = {
                            "cost_price": _to_float(p.get(prod_cost_col)),
                            "cost_includes_tax": bool(p.get(prod_cost_tax_col)),
                            "vat_rate": _to_float(p.get(prod_vat_col)),
                        }
            except Exception:
                pass

        # Atribuir costo a cada vendedor (misma fórmula que useSalesReports.ts)
        for item in raw_items:
            inv_id = item.get(fk_c)
            seller_id = invoice_to_seller.get(inv_id)
            if seller_id is None or seller_id not in agg:
                continue
            qty = _to_float(item.get(qty_c))
            raw_cost = item.get(purchase_cost_c)
            if raw_cost is not None and raw_cost != "":
                base_cost = _to_float(raw_cost)
                includes_tax_raw = item.get(purchase_cost_tax_c)
                includes_tax = bool(includes_tax_raw) if includes_tax_raw is not None else False
                vat_rate = _to_float(item.get(purchase_vat_c))
            else:
                prod_info = products_cost.get(item.get(product_id_c), {})
                base_cost = prod_info.get("cost_price", 0.0)
                includes_tax = prod_info.get("cost_includes_tax", False)
                vat_rate = prod_info.get("vat_rate", 0.0)
            tax_mult = 1.0 if includes_tax else (1.0 + vat_rate / 100.0)
            agg[seller_id]["costo_total"] += qty * base_cost * tax_mult

    # Paso 5: resolver nombres desde profiles
    profiles_t = _profiles_table()
    pid_c = _profiles_id_col()
    pname_c = _profiles_name_col()
    seller_ids = list(agg.keys())
    names: dict[Any, str] = {}
    if seller_ids:
        for i in range(0, len(seller_ids), 200):
            chunk = seller_ids[i : i + 200]
            r = client.table(profiles_t).select(f"{pid_c},{pname_c}").in_(pid_c, chunk).execute()
            for prow in r.data or []:
                names[prow.get(pid_c)] = str(prow.get(pname_c) or "")

    sort_key = "total_facturado" if metric != "cantidad" else "cantidad_facturas"
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][sort_key])
    out: list[dict[str, Any]] = []
    for sid, vals in ranked[:limit]:
        facturado = vals["total_facturado"]
        costo = round(vals["costo_total"], 2)
        utilidad = round(facturado - costo, 2)
        markup_pct = round((utilidad / costo) * 100, 2) if costo > 0 else 0.0
        pct_utilidad = round((utilidad / facturado) * 100, 2) if facturado > 0 else 0.0
        out.append({
            "user_id": sid,
            "nombre": names.get(sid) or str(sid),
            "total_facturado": round(facturado, 2),
            "costo_mercaderia": costo,
            "utilidad": utilidad,
            "markup_pct": markup_pct,
            "pct_utilidad_ventas": pct_utilidad,
            "cantidad_facturas": vals["cantidad_facturas"],
        })

    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "metrica": metric,
        "top_vendedores": out,
        "cantidad_devuelta": len(out),
        "limite": limit,
        "fuente": "supabase",
        "tabla": table,
        "nota_costo": "Costo de mercadería c/ IVA (purchase_cost del ítem o cost_price del producto). Markup = Utilidad / Costo × 100.",
    }


def _stub_top_customers_by_invoicing(desde: str, hasta: str, metric: str, limit: int) -> dict[str, Any]:
    demo = [
        {"customer_id": "demo-cust-1", "nombre": "Cliente Demo A", "total_facturado": 900000.0, "cantidad_facturas": 15},
        {"customer_id": "demo-cust-2", "nombre": "Cliente Demo B", "total_facturado": 650000.0, "cantidad_facturas": 10},
        {"customer_id": "demo-cust-3", "nombre": "Cliente Demo C", "total_facturado": 420000.0, "cantidad_facturas": 7},
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "metrica": metric,
        "top_clientes": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _top_customers_by_debt_from_supabase(limit: int) -> dict[str, Any]:
    """Devuelve los clientes con mayor saldo de deuda usando customer_balances_view."""
    client = _get_supabase()
    assert client is not None
    lim = max(1, min(limit, 50))

    # Paso 1: traer todos los saldos positivos ordenados por deuda DESC
    PAGE_SIZE = 1000
    all_balances: list[dict[str, Any]] = []
    start = 0
    while True:
        r = (
            client.from_("customer_balances_view")  # type: ignore[arg-type]
            .select("customer_id,total_debt,overdue_debt,current_debt,pending_invoices_count")
            .gt("total_debt", 0)
            .order("total_debt", desc=True)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        rows = r.data or []
        all_balances.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        if start > 100_000:
            break

    top_balances = all_balances[:lim]
    if not top_balances:
        return {"top_clientes": [], "cantidad_devuelta": 0, "fuente": "customer_balances_view"}

    # Paso 2: enriquecer con nombre y CUIT
    customer_ids = [row["customer_id"] for row in top_balances]
    table, id_c, name_c, tax_c, *_ = _customer_column_config()
    customer_map: dict[str, dict[str, str]] = {}
    CHUNK = 50
    for i in range(0, len(customer_ids), CHUNK):
        chunk = customer_ids[i: i + CHUNK]
        cr = (
            client.table(table)
            .select(f"{id_c},{name_c},{tax_c}")
            .in_(id_c, chunk)
            .execute()
        )
        for cust in (cr.data or []):
            customer_map[str(cust[id_c])] = {
                "nombre": str(cust.get(name_c) or ""),
                "tax_id": str(cust.get(tax_c) or ""),
            }

    result_rows = []
    for i, row in enumerate(top_balances, start=1):
        cid = row["customer_id"]
        info = customer_map.get(cid, {"nombre": "", "tax_id": ""})
        result_rows.append({
            "posicion": i,
            "customer_id": cid,
            "nombre": info["nombre"],
            "tax_id": info["tax_id"],
            "total_debt": round(float(row.get("total_debt") or 0), 2),
            "overdue_debt": round(float(row.get("overdue_debt") or 0), 2),
            "current_debt": round(float(row.get("current_debt") or 0), 2),
            "pending_invoices_count": int(row.get("pending_invoices_count") or 0),
        })

    return {
        "top_clientes": result_rows,
        "cantidad_devuelta": len(result_rows),
        "limite": lim,
        "fuente": "customer_balances_view",
    }


def _top_customers_by_invoicing_from_supabase(desde: str, hasta: str, metric: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    date_c = _invoices_date_column()
    amount_c = _invoices_amount_col()
    customer_id_c = _invoices_customer_id_column()

    # Paso 1: IDs que son destino de conversión (evitar doble conteo remito→factura)
    page = 1000
    excluded_ids: set[str] = set()
    try:
        excl_start = 0
        while True:
            excl_r = (
                client.table(table)
                .select("converted_to_invoice_id")
                .not_.is_("converted_to_invoice_id", "null")
                .order("id")
                .range(excl_start, excl_start + page - 1)
                .execute()
            )
            excl_rows = excl_r.data or []
            for row in excl_rows:
                if row.get("converted_to_invoice_id"):
                    excluded_ids.add(row["converted_to_invoice_id"])
            if len(excl_rows) < page:
                break
            excl_start += page
            if excl_start > 500_000:
                break
    except Exception:
        pass

    # Paso 2: paginar facturas con filtros (igual que validInvoices del front)
    excluded_statuses = ["cancelled", "voided", "converted", "draft"]
    excluded_types = {"nota_pedido", "np"}
    desde_filtro, hasta_filtro = _date_range_bounds(desde, hasta)
    start = 0
    inv_rows: list[dict[str, Any]] = []
    while True:
        r = (
            client.table(table)
            .select(f"id,{amount_c},{customer_id_c},warehouse_id,sales_order_id,payment_condition,invoice_type")
            .gte(date_c, desde_filtro)
            .lte(date_c, hasta_filtro)
            .not_.in_("status", excluded_statuses)
            .order("id")
            .range(start, start + page - 1)
            .execute()
        )
        rows: list[dict[str, Any]] = r.data or []
        for row in rows:
            if row.get("id") in excluded_ids:
                continue
            if (row.get("invoice_type") or "").lower() in excluded_types:
                continue
            if not (row.get("warehouse_id") or row.get("sales_order_id") or row.get("payment_condition")):
                continue
            inv_rows.append(row)
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    # Paso 3: agregar por cliente
    agg: dict[Any, dict[str, float]] = {}
    for row in inv_rows:
        cid = row.get(customer_id_c)
        if cid is None:
            continue
        cur = agg.setdefault(cid, {"total_facturado": 0.0, "cantidad_facturas": 0})
        cur["total_facturado"] += _to_float(row.get(amount_c))
        cur["cantidad_facturas"] += 1

    # Paso 4: resolver nombres de clientes
    cust_table, id_c, name_c, *_ = _customer_column_config()
    customer_ids = list(agg.keys())
    names: dict[Any, str] = {}
    chunk_sz = 200
    for i in range(0, len(customer_ids), chunk_sz):
        chunk = customer_ids[i : i + chunk_sz]
        try:
            cr = client.table(cust_table).select(f"{id_c},{name_c}").in_(id_c, chunk).execute()
            for crow in cr.data or []:
                names[crow.get(id_c)] = str(crow.get(name_c) or "")
        except Exception:
            pass

    sort_key = "total_facturado" if metric != "cantidad" else "cantidad_facturas"
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][sort_key])
    out: list[dict[str, Any]] = []
    for cid, vals in ranked[:limit]:
        out.append({
            "customer_id": cid,
            "nombre": names.get(cid) or str(cid),
            "total_facturado": round(vals["total_facturado"], 2),
            "cantidad_facturas": vals["cantidad_facturas"],
        })

    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "metrica": metric,
        "top_clientes": out,
        "cantidad_devuelta": len(out),
        "limite": limit,
        "fuente": "supabase",
        "tabla": table,
    }


def _stub_top_sellers(desde: str, hasta: str, metric: str, limit: int) -> dict[str, Any]:
    demo = [
        {"user_id": "demo-user-1", "nombre": "Ana Gómez", "total_monto": 850000.0, "cantidad_ordenes": 12},
        {"user_id": "demo-user-2", "nombre": "Carlos Pérez", "total_monto": 620000.0, "cantidad_ordenes": 9},
        {"user_id": "demo-user-3", "nombre": "María López", "total_monto": 410000.0, "cantidad_ordenes": 7},
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "metrica": metric,
        "top_vendedores": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
        "nota": "definí SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY para leer Postgres",
    }


def _top_sellers_from_supabase(desde: str, hasta: str, metric: str, limit: int) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _orders_list_table()
    date_c = _orders_list_date_column()
    seller_c = _orders_seller_col()
    amount_c = (os.environ.get("ERP_SUPABASE_SALES_AMOUNT_COL") or "total_amount").strip() or "total_amount"

    page = 1000
    start = 0
    agg: dict[Any, dict[str, float]] = {}

    while True:
        r = (
            client.table(table)
            .select(f"{seller_c},{amount_c}")
            .gte(date_c, desde)
            .lte(date_c, hasta)
            .range(start, start + page - 1)
            .execute()
        )
        rows: list[dict[str, Any]] = r.data or []
        for row in rows:
            sid = row.get(seller_c)
            if sid is None:
                continue
            cur = agg.setdefault(sid, {"total_monto": 0.0, "cantidad_ordenes": 0})
            cur["total_monto"] += _to_float(row.get(amount_c))
            cur["cantidad_ordenes"] += 1
        if len(rows) < page:
            break
        start += page
        if start > 500_000:
            break

    profiles_t = _profiles_table()
    pid_c = _profiles_id_col()
    pname_c = _profiles_name_col()
    seller_ids = list(agg.keys())
    names: dict[Any, str] = {}
    if seller_ids:
        for i in range(0, len(seller_ids), 200):
            chunk = seller_ids[i : i + 200]
            r = client.table(profiles_t).select(f"{pid_c},{pname_c}").in_(pid_c, chunk).execute()
            for prow in r.data or []:
                names[prow.get(pid_c)] = str(prow.get(pname_c) or "")

    sort_key = "total_monto" if metric != "cantidad" else "cantidad_ordenes"
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][sort_key])
    out: list[dict[str, Any]] = []
    for sid, vals in ranked[:limit]:
        out.append({
            "user_id": sid,
            "nombre": names.get(sid) or str(sid),
            "total_monto": round(vals["total_monto"], 2),
            "cantidad_ordenes": vals["cantidad_ordenes"],
        })

    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "metrica": metric,
        "top_vendedores": out,
        "cantidad_devuelta": len(out),
        "limite": limit,
        "fuente": "supabase",
        "tabla": table,
    }


def dispatch_tool(name: str, arguments_json: str) -> str:
    args: dict[str, Any] = {}
    if arguments_json:
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "argumentos JSON inválidos"})

    use_sb = _get_supabase() is not None

    try:
        if name == "get_sales_summary":
            desde, hasta = str(args["desde"]), str(args["hasta"])
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", desde) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", hasta
            ):
                return json.dumps(
                    {"error": "fechas deben ser YYYY-MM-DD"},
                    ensure_ascii=False,
                )
            if use_sb:
                result = _sales_from_supabase(desde, hasta)
            else:
                result = _stub_sales(desde, hasta)
        elif name == "search_products":
            lim = _coerce_limit(args.get("limit", 10))
            result = (
                _products_from_supabase(
                    str(args["query"]),
                    lim,
                )
                if use_sb
                else _stub_products(
                    str(args["query"]),
                    lim,
                )
            )
        elif name == "search_suppliers":
            lim = _coerce_limit(args.get("limit", 10))
            result = (
                _suppliers_from_supabase(str(args["query"]), lim)
                if use_sb
                else _stub_suppliers(str(args["query"]), lim)
            )
        elif name == "search_customers":
            lim = _coerce_limit(args.get("limit", 10))
            result = (
                _customers_from_supabase(str(args["query"]), lim)
                if use_sb
                else _stub_customers(str(args["query"]), lim)
            )
        elif name == "get_customer_balance":
            customer_arg = str(args["customer"]).strip()
            if use_sb:
                result = _customer_balance_from_supabase(customer_arg)
            else:
                # stub: devuelve datos de ejemplo
                result = {
                    "customer_id": "demo-id",
                    "nombre": customer_arg,
                    "total_debt": 150000.0,
                    "total_debt_bn1": 90000.0,
                    "total_debt_bn2": 60000.0,
                    "overdue_debt": 50000.0,
                    "current_debt": 100000.0,
                    "pending_invoices_count": 3,
                    "situacion": "deuda",
                    "fuente": "stub",
                }
        elif name == "list_sales_orders":
            lim = _coerce_limit(args.get("limit"), default=30, cap=200)
            try:
                desde, hasta = _resolve_orders_list_dates(
                    args.get("desde"),
                    args.get("hasta"),
                )
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            filter_cid: str | None = None
            if use_sb:
                cid_raw = str(args.get("customer_id") or "").strip()
                cname_raw = str(args.get("customer_name") or "").strip()
                if cid_raw:
                    filter_cid = cid_raw
                elif cname_raw:
                    filter_cid = _resolve_customer_id_by_name(cname_raw)
                    if not filter_cid:
                        return json.dumps({"error": f"No se encontró cliente: {cname_raw!r}"}, ensure_ascii=False)
                result = _list_sales_orders_from_supabase(desde, hasta, lim, filter_cid)
            else:
                result = _stub_list_sales_orders(desde, hasta, lim)
        elif name == "count_sales_orders_by_status":
            try:
                desde, hasta = _resolve_orders_list_dates(
                    args.get("desde"),
                    args.get("hasta"),
                )
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _count_sales_orders_by_status_from_supabase(desde, hasta)
            else:
                result = _stub_count_sales_orders_by_status(desde, hasta)
        elif name == "get_top_selling_products":
            lim = _coerce_limit(args.get("limit"), default=10, cap=100)
            try:
                desde, hasta = _resolve_orders_list_dates(
                    args.get("desde"),
                    args.get("hasta"),
                )
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _top_selling_products_from_supabase(desde, hasta, lim)
            else:
                result = _stub_top_selling_products(desde, hasta, lim)
        elif name == "get_least_selling_products":
            lim = _coerce_limit(args.get("limit"), default=10, cap=100)
            try:
                desde, hasta = _resolve_orders_list_dates(
                    args.get("desde"),
                    args.get("hasta"),
                )
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _least_selling_products_from_supabase(desde, hasta, lim)
            else:
                result = _stub_least_selling_products(desde, hasta, lim)
        elif name == "get_purchase_summary":
            desde, hasta = str(args["desde"]), str(args["hasta"])
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", desde) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", hasta
            ):
                return json.dumps(
                    {"error": "fechas deben ser YYYY-MM-DD"},
                    ensure_ascii=False,
                )
            if use_sb:
                result = _purchase_summary_from_supabase(desde, hasta)
            else:
                result = _stub_purchase_summary(desde, hasta)
        elif name == "list_purchase_orders":
            lim = _coerce_limit(args.get("limit"), default=30, cap=200)
            try:
                desde, hasta = _resolve_po_list_dates(
                    args.get("desde"),
                    args.get("hasta"),
                )
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _list_purchase_orders_from_supabase(desde, hasta, lim)
            else:
                result = _stub_list_purchase_orders(desde, hasta, lim)
        elif name == "count_purchase_orders_by_status":
            try:
                desde, hasta = _resolve_po_list_dates(
                    args.get("desde"),
                    args.get("hasta"),
                )
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _count_purchase_orders_by_status_from_supabase(desde, hasta)
            else:
                result = _stub_count_purchase_orders_by_status(desde, hasta)
        elif name == "list_purchase_order_items":
            po_id = str(args["purchase_order_id"]).strip()
            if not _is_uuid(po_id):
                return json.dumps(
                    {"error": "purchase_order_id debe ser un UUID válido"},
                    ensure_ascii=False,
                )
            lim = _coerce_limit(args.get("limit"), default=100, cap=500)
            if use_sb:
                result = _list_purchase_order_items_from_supabase(po_id, lim)
            else:
                result = _stub_list_purchase_order_items(po_id, lim)
        elif name == "get_product_available_stock":
            lim = _coerce_limit(args.get("limit"), default=5, cap=20)
            if use_sb:
                result = _product_available_stock_from_supabase(str(args["query"]), lim)
            else:
                result = _stub_product_available_stock(str(args["query"]), lim)
        elif name == "list_products_below_min_stock":
            lim = _coerce_limit(args.get("limit"), default=50, cap=200)
            if use_sb:
                result = _products_below_min_stock_from_supabase(lim)
            else:
                result = _stub_products_below_min_stock(lim)
        elif name == "list_top_products_by_stock":
            lim = _coerce_limit(args.get("limit"), default=10, cap=100)
            order = str(args.get("order") or "desc").strip().lower()
            if order not in ("asc", "desc"):
                order = "desc"
            wh_name = str(args.get("warehouse_name") or "").strip() or None
            only_principal = bool(args.get("only_principal") or False)
            if use_sb:
                result = _top_products_by_stock_from_supabase(lim, order, wh_name, only_principal)
            else:
                result = _stub_top_products_by_stock(lim, order)
        elif name == "list_recent_product_movements":
            lim = _coerce_limit(args.get("limit"), default=20, cap=100)
            days = _coerce_limit(args.get("days"), default=7, cap=90)
            if use_sb:
                result = _recent_product_movements_from_supabase(str(args["query"]), days, lim)
            else:
                result = _stub_recent_product_movements(str(args["query"]), days, lim)
        elif name == "list_sales_order_items":
            lim = _coerce_limit(args.get("limit"), default=100, cap=500)
            oid = str(args.get("sales_order_id") or "").strip()
            order_num = str(args.get("order_number") or "").strip()
            if not oid and not order_num:
                return json.dumps({"error": "Requerido: sales_order_id (UUID) u order_number"}, ensure_ascii=False)
            if not oid or not _is_uuid(oid):
                if use_sb and order_num:
                    resolved = _resolve_sales_order_id_by_number(order_num)
                    if not resolved:
                        return json.dumps({"error": f"No se encontró la orden {order_num!r}"}, ensure_ascii=False)
                    oid = resolved
                elif not use_sb:
                    oid = oid or order_num
                else:
                    return json.dumps({"error": "sales_order_id debe ser un UUID válido, o proveer order_number"}, ensure_ascii=False)
            if use_sb:
                result = _list_sales_order_items_from_supabase(oid, lim)
            else:
                result = _stub_list_sales_order_items(oid, lim)
        elif name == "list_customer_invoices":
            lim = _coerce_limit(args.get("limit"), default=10, cap=200)
            try:
                desde, hasta = _resolve_orders_list_dates(args.get("desde"), args.get("hasta"))
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            filter_cid: str | None = None
            if use_sb:
                cid_raw = str(args.get("customer_id") or "").strip()
                cname_raw = str(args.get("customer_name") or "").strip()
                if cid_raw:
                    filter_cid = cid_raw
                elif cname_raw:
                    filter_cid = _resolve_customer_id_by_name(cname_raw)
                    if not filter_cid:
                        return json.dumps({"error": f"No se encontró cliente: {cname_raw!r}"}, ensure_ascii=False)
                status_raw = str(args.get("status") or "").strip() or None
                result = _list_customer_invoices_from_supabase(desde, hasta, lim, filter_cid, status_raw)
            else:
                result = _stub_list_customer_invoices(desde, hasta, lim)
        elif name == "get_invoice_summary":
            desde, hasta = str(args["desde"]), str(args["hasta"])
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", desde) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", hasta):
                return json.dumps({"error": "fechas deben ser YYYY-MM-DD"}, ensure_ascii=False)
            filter_cid = None
            if use_sb:
                cid_raw = str(args.get("customer_id") or "").strip()
                cname_raw = str(args.get("customer_name") or "").strip()
                if cid_raw:
                    filter_cid = cid_raw
                elif cname_raw:
                    filter_cid = _resolve_customer_id_by_name(cname_raw)
                    if not filter_cid:
                        return json.dumps({"error": f"No se encontró cliente: {cname_raw!r}"}, ensure_ascii=False)
                result = _get_invoice_summary_from_supabase(desde, hasta, filter_cid)
            else:
                result = _stub_get_invoice_summary(desde, hasta)
        elif name == "get_profit_margin_summary":
            desde, hasta = str(args["desde"]), str(args["hasta"])
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", desde) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", hasta):
                return json.dumps({"error": "fechas deben ser YYYY-MM-DD"}, ensure_ascii=False)
            filter_cid = None
            if use_sb:
                cid_raw = str(args.get("customer_id") or "").strip()
                cname_raw = str(args.get("customer_name") or "").strip()
                if cid_raw:
                    filter_cid = cid_raw
                elif cname_raw:
                    filter_cid = _resolve_customer_id_by_name(cname_raw)
                    if not filter_cid:
                        return json.dumps({"error": f"No se encontró cliente: {cname_raw!r}"}, ensure_ascii=False)
                result = _get_profit_margin_from_supabase(desde, hasta, filter_cid)
            else:
                result = _stub_get_profit_margin(desde, hasta)
        elif name == "list_customer_invoice_items":
            lim = _coerce_limit(args.get("limit"), default=100, cap=500)
            inv_id = str(args.get("invoice_id") or "").strip()
            inv_num = str(args.get("invoice_number") or "").strip()
            if not inv_id and not inv_num:
                return json.dumps({"error": "Requerido: invoice_id (UUID) o invoice_number"}, ensure_ascii=False)
            if not inv_id or not _is_uuid(inv_id):
                if use_sb and inv_num:
                    resolved = _resolve_invoice_id_by_number(inv_num)
                    if not resolved:
                        return json.dumps({"error": f"No se encontró la factura {inv_num!r}"}, ensure_ascii=False)
                    inv_id = resolved
                elif not use_sb:
                    inv_id = inv_id or inv_num
                else:
                    return json.dumps({"error": "invoice_id debe ser UUID válido, o proveer invoice_number"}, ensure_ascii=False)
            if use_sb:
                result = _list_customer_invoice_items_from_supabase(inv_id, lim)
            else:
                result = _stub_list_customer_invoice_items(inv_id, lim)
        elif name == "list_customer_payments":
            lim = _coerce_limit(args.get("limit"), default=50, cap=200)
            inv_id = str(args.get("invoice_id") or "").strip()
            inv_num = str(args.get("invoice_number") or "").strip()
            desde_p = str(args.get("desde") or "").strip() or None
            hasta_p = str(args.get("hasta") or "").strip() or None
            filter_cid = None
            if use_sb:
                if inv_num and (not inv_id or not _is_uuid(inv_id)):
                    resolved = _resolve_invoice_id_by_number(inv_num)
                    if not resolved:
                        return json.dumps({"error": f"No se encontró la factura {inv_num!r}"}, ensure_ascii=False)
                    inv_id = resolved
                cid_raw = str(args.get("customer_id") or "").strip()
                cname_raw = str(args.get("customer_name") or "").strip()
                if cid_raw:
                    filter_cid = cid_raw
                elif cname_raw:
                    filter_cid = _resolve_customer_id_by_name(cname_raw)
                    if not filter_cid:
                        return json.dumps({"error": f"No se encontró cliente: {cname_raw!r}"}, ensure_ascii=False)
                if not inv_id and not filter_cid and not desde_p:
                    return json.dumps({"error": "Requerido: invoice_id, invoice_number, customer_name/id, o rango de fechas"}, ensure_ascii=False)
                result = _list_customer_payments_from_supabase(
                    lim,
                    filter_invoice_id=inv_id or None,
                    filter_customer_id=filter_cid,
                    desde=desde_p,
                    hasta=hasta_p,
                )
            else:
                result = _stub_list_customer_payments(lim)
        elif name == "list_products_without_stock_days":
            days = _coerce_limit(args.get("days"), default=15, cap=365)
            lim = _coerce_limit(args.get("limit"), default=50, cap=200)
            if use_sb:
                result = _products_without_stock_days_from_supabase(days, lim)
            else:
                result = _stub_products_without_stock_days(days, lim)
        elif name == "get_top_sellers":
            lim = _coerce_limit(args.get("limit"), default=10, cap=50)
            metric = str(args.get("metric") or "monto").strip().lower()
            if metric not in ("monto", "cantidad"):
                metric = "monto"
            try:
                desde, hasta = _resolve_orders_list_dates(args.get("desde"), args.get("hasta"))
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _top_sellers_from_supabase(desde, hasta, metric, lim)
            else:
                result = _stub_top_sellers(desde, hasta, metric, lim)
        elif name == "get_top_sellers_by_invoicing":
            lim = _coerce_limit(args.get("limit"), default=10, cap=50)
            metric = str(args.get("metric") or "monto").strip().lower()
            if metric not in ("monto", "cantidad"):
                metric = "monto"
            # Default: mes actual (1° del mes → hoy) en lugar de los últimos 90 días
            _desde_raw = args.get("desde")
            _hasta_raw = args.get("hasta")
            if not _desde_raw and not _hasta_raw:
                _today = date.today()
                _desde_raw = _today.replace(day=1).isoformat()
                _hasta_raw = _today.isoformat()
            try:
                desde, hasta = _resolve_orders_list_dates(_desde_raw, _hasta_raw)
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            # Tope de 62 días para evitar timeouts (el embedded select es eficiente pero tiene límites)
            if (date.fromisoformat(hasta) - date.fromisoformat(desde)).days > 62:
                return json.dumps({
                    "error": (
                        f"El período solicitado ({desde} → {hasta}) supera los 62 días permitidos. "
                        "Usá un rango más corto, por ejemplo el mes actual o un mes específico."
                    )
                }, ensure_ascii=False)
            if use_sb:
                result = _top_sellers_by_invoicing_from_supabase(desde, hasta, metric, lim)
            else:
                result = _stub_top_sellers_by_invoicing(desde, hasta, metric, lim)
        elif name == "get_top_customers_by_invoicing":
            lim = _coerce_limit(args.get("limit"), default=10, cap=50)
            metric = str(args.get("metric") or "monto").strip().lower()
            if metric not in ("monto", "cantidad"):
                metric = "monto"
            try:
                desde, hasta = _resolve_orders_list_dates(args.get("desde"), args.get("hasta"))
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _top_customers_by_invoicing_from_supabase(desde, hasta, metric, lim)
            else:
                result = _stub_top_customers_by_invoicing(desde, hasta, metric, lim)
        elif name == "get_top_customers_by_debt":
            lim = _coerce_limit(args.get("limit"), default=10, cap=50)
            if use_sb:
                result = _top_customers_by_debt_from_supabase(lim)
            else:
                result = {
                    "top_clientes": [
                        {"posicion": 1, "nombre": "Cliente Demo A", "tax_id": "20-00000001-0", "total_debt": 500000.0, "overdue_debt": 300000.0, "current_debt": 200000.0, "pending_invoices_count": 5},
                        {"posicion": 2, "nombre": "Cliente Demo B", "tax_id": "20-00000002-0", "total_debt": 320000.0, "overdue_debt": 120000.0, "current_debt": 200000.0, "pending_invoices_count": 3},
                    ],
                    "cantidad_devuelta": 2,
                    "fuente": "stub",
                }
        elif name == "get_product_sales_units":
            query = str(args["query"]).strip()
            try:
                desde, hasta = _resolve_orders_list_dates(args.get("desde"), args.get("hasta"))
            except ValueError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
            if use_sb:
                result = _get_product_sales_units_from_supabase(query, desde, hasta)
            else:
                result = _stub_get_product_sales_units(query, desde, hasta)
        else:
            return json.dumps({"error": f"tool desconocida: {name}"})
        return json.dumps(result, ensure_ascii=False)
    except KeyError as e:
        return json.dumps({"error": f"falta parámetro: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {"error": str(e), "hint": "revisá tablas/columnas o RLS en Supabase"},
            ensure_ascii=False,
        )
