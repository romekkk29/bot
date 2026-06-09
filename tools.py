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
from datetime import date, datetime, timedelta
from typing import Any

# Schemas que entiende Groq (mismo formato que OpenAI tool calling)
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": (
                "Obtiene un resumen agregado de ventas (totales) para un rango YYYY-MM-DD. "
                "NO devuelve el detalle de cada orden. "
                "Para listar órdenes de venta usar list_sales_orders. "
                "Usar cuando pregunten por facturación total del período, montos sumados."
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
                "Lista órdenes de venta (número, fecha, total, estado, etc.) en un rango de fechas. "
                "Acepta filtro opcional por cliente (nombre o UUID). "
                "Si el usuario menciona un cliente específico, pasar customer_name. "
                "Si pide 'todas las ventas' o 'todo el detalle' sin filtro temporal, NO llamar: pedirle al usuario que acoter el período. "
                "Usar cuando pregunten qué órdenes hay, pedidos de venta, OV, listado de ventas, ventas de un cliente."
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
                "Obtiene top productos vendidos en un rango de fechas a partir de líneas de venta. "
                "Devuelve ranking por cantidad y monto vendido por producto."
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
                "Consulta stock disponible de un producto por nombre/código/SKU/barcode. "
                "Resuelve el producto en products y luego calcula stock disponible desde warehouse_stock."
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
                "Lista facturas de clientes (número, fecha, total, cobrado, saldo, estado) en un rango de fechas. "
                "Acepta filtro opcional por cliente (nombre o UUID) y/o estado. "
                "Usar cuando pregunten por facturas, comprobantes emitidos, deuda de un cliente, facturas pendientes/vencidas."
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
                        "description": "Máximo de facturas a devolver (default 30, máx 200)",
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
                "Resumen agregado de facturación en un rango de fechas: total facturado, total cobrado, saldo pendiente, "
                "cantidad de facturas y desglose por estado. "
                "Acepta filtro opcional por cliente. "
                "Usar cuando pregunten cuánto facturamos, cuánto cobró un cliente, saldo total de deuda."
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
]

_supabase_client: Any | None = None


def data_backend_label() -> str:
    if _supabase_credentials():
        return "supabase"
    return "stub"


def _supabase_credentials() -> tuple[str, str] | None:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_KEY", "").strip()
    )
    if url and key:
        return (url, key)
    return None


def _get_supabase():
    global _supabase_client
    creds = _supabase_credentials()
    if not creds:
        return None
    if _supabase_client is None:
        from supabase import create_client

        url, key = creds
        _supabase_client = create_client(url, key)
    return _supabase_client


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
    q = client.table(table).select(select_cols).gte(date_c, desde).lte(date_c, hasta)
    if filter_customer_id:
        q = q.eq(customer_id_c, filter_customer_id)
    if filter_status:
        q = q.eq("status", filter_status)
    r = q.order(date_c, desc=True).limit(lim).execute()
    rows: list[dict[str, Any]] = r.data or []
    rows = _attach_customers_to_orders(rows, customer_id_c)
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "facturas": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": table,
    }


def _get_invoice_summary_from_supabase(
    desde: str, hasta: str,
    filter_customer_id: str | None = None,
) -> dict[str, Any]:
    client = _get_supabase()
    assert client is not None
    table = _invoices_table()
    date_c = _invoices_date_column()
    customer_id_c = _invoices_customer_id_column()
    q = client.table(table).select(
        "total_amount,paid_amount,remaining_amount,status"
    ).gte(date_c, desde).lte(date_c, hasta)
    if filter_customer_id:
        q = q.eq(customer_id_c, filter_customer_id)
    r = q.limit(5000).execute()
    rows: list[dict[str, Any]] = r.data or []
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
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "ordenes": rows,
        "cantidad_devuelta": len(rows),
        "limite": lim,
        "fuente": "supabase",
        "tabla": table,
    }


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
        },
        {
            "product_id": "demo-prod-2",
            "sku_o_codigo": "SKU-002",
            "nombre": "Producto demo B",
            "cantidad_vendida": 95.0,
            "monto_vendido": 330000.0,
        },
    ]
    return {
        "periodo": {"desde": desde, "hasta": hasta},
        "top_productos": demo[:limit],
        "cantidad_devuelta": min(len(demo), limit),
        "limite": limit,
        "fuente": "stub",
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
    if uuid_c and ids:
        for i in range(0, len(ids), 200):
            chunk = ids[i : i + 200]
            r = client.table(_products_table()).select(f"{uuid_c},{code_c},{name_c}").in_(uuid_c, chunk).execute()
            for prow in r.data or []:
                by_id[prow.get(uuid_c)] = prow

    ranked = sorted(
        agg.items(),
        key=lambda kv: (-kv[1]["cantidad_vendida"], -kv[1]["monto_vendido"]),
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
        "top_productos": out,
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
    r = client.table(table).select(amount_c).gte(date_c, desde).lte(date_c, hasta).execute()
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
    r = q.gte(date_c, desde).lte(date_c, hasta).execute()
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
        oq = client.table(orders_t).select(oid_c).gte(date_c, desde).lte(date_c, hasta)
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
            lim = _coerce_limit(args.get("limit"), default=30, cap=200)
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
