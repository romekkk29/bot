module.exports = {
  apps: [
    {
      name: "erp-bot",
      script: "whatsapp_bot.py",
      interpreter: "/opt/bot/venv/bin/python",
      cwd: "/opt/bot",
      env_file: "/opt/bot/.env",
      restart_delay: 5000,
      max_restarts: 10,
      watch: false,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};
