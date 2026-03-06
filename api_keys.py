"""
API keys loader for DeepLens.

This module exposes API key variables. It will try to import a user-provided
`api_keys.py` (not tracked in source control). If that file is missing it will
fall back to the `api_keys_template.py` defaults and emit a warning to the logger.

To provide real keys, copy `config/api_keys_template.py` -> `config/api_keys.py`
and replace the placeholders. Do not commit `config/api_keys.py`.
"""
import logging

logger = logging.getLogger("config.api_keys")

try:
	# If the user created a real `config/api_keys.py`, prefer it
	from .api_keys import *  # type: ignore
	logger.info("Loaded API keys from config.api_keys")
except Exception:
	# Fall back to template defaults
	from .api_keys_template import *  # noqa: F401,F403
	logger.warning(
		"config.api_keys not found; using api_keys_template defaults. "
		"Copy api_keys_template.py to api_keys.py and add real keys to avoid this warning."
	)