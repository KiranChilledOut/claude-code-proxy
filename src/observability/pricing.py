import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    model: str
    input_per_1m: float
    output_per_1m: float
    advertised_tok_s: Optional[float] = None
    currency: str = "USD"


class PricingCatalog:
    """Env-configured pricing lookup for backend model usage."""

    def __init__(self, raw_json: str):
        self.raw_json = raw_json or "{}"
        self.prices = self._load_prices(self.raw_json)

    def quote(self, model: Optional[str], input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        price = self.get(model)
        if price is None:
            return {
                "input_cost": None,
                "output_cost": None,
                "estimated_cost": None,
                "currency": None,
                "advertised_tok_s": None,
            }

        input_cost = (input_tokens / 1_000_000) * price.input_per_1m
        output_cost = (output_tokens / 1_000_000) * price.output_per_1m
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "estimated_cost": input_cost + output_cost,
            "currency": price.currency,
            "advertised_tok_s": price.advertised_tok_s,
        }

    def get(self, model: Optional[str]) -> Optional[ModelPrice]:
        if not model:
            return self.prices.get("default") or self.prices.get("*")
        return self.prices.get(model) or self.prices.get("default") or self.prices.get("*")

    def as_list(self) -> list:
        return [
            {
                "model": price.model,
                "input_per_1m": price.input_per_1m,
                "output_per_1m": price.output_per_1m,
                "advertised_tok_s": price.advertised_tok_s,
                "currency": price.currency,
            }
            for price in self.prices.values()
        ]

    def _load_prices(self, raw_json: str) -> Dict[str, ModelPrice]:
        try:
            parsed = json.loads(raw_json or "{}")
        except json.JSONDecodeError as exc:
            logger.warning("MODEL_PRICES_JSON is invalid JSON: %s", exc)
            return {}

        if not isinstance(parsed, dict):
            logger.warning("MODEL_PRICES_JSON must be a JSON object")
            return {}

        prices: Dict[str, ModelPrice] = {}
        for model, payload in parsed.items():
            if not isinstance(payload, dict):
                continue

            input_price = _read_float(
                payload,
                "input_per_1m",
                "input_per_million",
                "in_per_1m",
                "in",
            )
            output_price = _read_float(
                payload,
                "output_per_1m",
                "output_per_million",
                "out_per_1m",
                "out",
            )
            if input_price is None or output_price is None:
                logger.warning("Skipping price for %s: missing input/output price", model)
                continue

            prices[model] = ModelPrice(
                model=model,
                input_per_1m=input_price,
                output_per_1m=output_price,
                advertised_tok_s=_read_float(
                    payload, "advertised_tok_s", "tok_s", "tokens_per_second"
                ),
                currency=str(payload.get("currency") or "USD"),
            )

        return prices


def _read_float(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key not in payload:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            return None
    return None
