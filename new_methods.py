    @staticmethod
    def send_email_batch(
        alerts: list[dict],
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> bool:
        if not alerts:
            return False
        import json
        from datetime import datetime
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        intent = {
            __orchestrator_intent__: email,
            subject: subject,
            alerts: list(alerts)
        }
        print(json.dumps(intent, default=default_serializer), flush=True)
        return True

    @staticmethod
    def send_phone_webhook_batch(alerts, webhook_url: Optional[str] = None):
        if not alerts:
            return False
        import json
        from datetime import datetime
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        intent = {
            __orchestrator_intent__: ntfy_webhook,
            webhook_url: webhook_url,
            alerts: list(alerts)
        }
        print(json.dumps(intent, default=default_serializer), flush=True)
        return True

    @staticmethod
    def _send_urgent_email_fallback(alerts) -> None:
        pass

    @staticmethod
    def _send_budget_warning(channel: str, reason: str) -> None:
        pass
