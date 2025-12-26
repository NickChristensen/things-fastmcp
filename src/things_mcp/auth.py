"""
Simple in-memory OAuth 2.1 Authorization Server Provider for Things FastMCP.
Implements the OAuthAuthorizationServerProvider protocol for ChatGPT integration.
"""
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode, urlparse

from pydantic import AnyUrl
from mcp.server.auth.provider import (
    OAuthAuthorizationServerProvider,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    AccessToken,
    AuthorizeError,
    TokenError,
    RegistrationError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .logging_config import get_logger

logger = get_logger(__name__)


class SimpleOAuthProvider(OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]):
    """
    Simple in-memory OAuth provider for MCP server.

    This provider:
    - Supports one pre-configured client (from environment variables)
    - Does NOT support Dynamic Client Registration
    - Stores authorization codes and tokens in memory
    - Issues opaque tokens (not JWTs)
    - Implements Authorization Code + PKCE flow
    """

    def __init__(self, client_id: str, client_secret: str, issuer_url: str):
        """
        Initialize the OAuth provider.

        Args:
            client_id: Pre-configured OAuth client ID
            client_secret: Pre-configured OAuth client secret
            issuer_url: The issuer URL for this OAuth server
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.issuer_url = issuer_url

        # In-memory storage
        self._authorization_codes: dict[str, AuthorizationCode] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}
        self._access_tokens: dict[str, AccessToken] = {}

        logger.info(f"SimpleOAuthProvider initialized with client_id={client_id[:8]}...")

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        """
        Retrieve pre-configured client information.

        Args:
            client_id: The client ID to look up

        Returns:
            Client information if the client_id matches our pre-configured client
        """
        if client_id != self.client_id:
            logger.warning(f"Client lookup failed: {client_id} != {self.client_id[:8]}...")
            return None

        # Return our pre-configured client
        # ChatGPT's redirect URIs include:
        # - https://chatgpt.com/connector_platform_oauth_redirect (MCP connectors)
        # - https://chat.openai.com/aip/{app-id}/oauth/callback (older format)
        # We'll accept any chat.openai.com or chatgpt.com redirect URI
        return OAuthClientInformationFull(
            client_id=self.client_id,
            client_secret=self.client_secret,
            grant_types=["authorization_code", "refresh_token"],
            redirect_uris=[
              AnyUrl("https://chat.openai.com/aip/"),
              AnyUrl("https://chatgpt.com/aip/"),
              AnyUrl("https://chat.openai.com/"),
              AnyUrl("https://chatgpt.com/"),
              AnyUrl("https://chatgpt.com/connector_platform_oauth_redirect")
            ],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
            client_name="ChatGPT MCP Client",
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """
        Dynamic Client Registration is NOT supported.

        Raises:
            RegistrationError: Always, since DCR is disabled
        """
        logger.warning("Dynamic Client Registration attempted but not supported")
        raise RegistrationError(
            error="invalid_client_metadata",
            error_description="Dynamic Client Registration is not supported. Use pre-configured client credentials."
        )

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams
    ) -> str:
        """
        Handle authorization request and return redirect URL.

        In a full implementation, this would redirect to a user consent page.
        For simplicity, we'll auto-approve and generate an authorization code.

        Args:
            client: The requesting client
            params: Authorization parameters

        Returns:
            Redirect URL with authorization code

        Raises:
            AuthorizeError: If the request is invalid
        """
        logger.info(f"Authorization request from client {client.client_id[:8]}...")

        # Validate redirect URI
        redirect_uri_str = str(params.redirect_uri)
        valid_redirect = False
        for allowed_uri in client.redirect_uris:
            if redirect_uri_str.startswith(str(allowed_uri)):
                valid_redirect = True
                break

        if not valid_redirect:
            logger.error(f"Invalid redirect_uri: {redirect_uri_str}")
            raise AuthorizeError(
                error="invalid_request",
                error_description=f"Invalid redirect_uri: {redirect_uri_str}"
            )

        # Generate authorization code (160+ bits of entropy as recommended)
        code = secrets.token_urlsafe(32)  # 256 bits

        # Store authorization code
        auth_code = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 600,  # 10 minutes
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        self._authorization_codes[code] = auth_code

        logger.info(f"Generated authorization code for client {client.client_id[:8]}...")

        # Build redirect URL with authorization code
        redirect_params = {"code": code}
        if params.state:
            redirect_params["state"] = params.state

        redirect_url = f"{params.redirect_uri}?{urlencode(redirect_params)}"
        return redirect_url

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str
    ) -> Optional[AuthorizationCode]:
        """
        Load an authorization code.

        Args:
            client: The client requesting the code
            authorization_code: The code to load

        Returns:
            The AuthorizationCode object, or None if not found/expired
        """
        code_obj = self._authorization_codes.get(authorization_code)

        if not code_obj:
            logger.warning(f"Authorization code not found: {authorization_code[:8]}...")
            return None

        # Verify it belongs to this client
        if code_obj.client_id != client.client_id:
            logger.warning(f"Authorization code client mismatch")
            return None

        # Check expiration
        if code_obj.expires_at < time.time():
            logger.warning(f"Authorization code expired")
            del self._authorization_codes[authorization_code]
            return None

        return code_obj

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode
    ) -> OAuthToken:
        """
        Exchange authorization code for access and refresh tokens.

        Args:
            client: The client exchanging the code
            authorization_code: The authorization code to exchange

        Returns:
            OAuth token with access_token and refresh_token

        Raises:
            TokenError: If the exchange fails
        """
        logger.info(f"Exchanging authorization code for tokens (client={client.client_id[:8]}...)")

        # Generate access token
        access_token_str = secrets.token_urlsafe(32)
        access_token = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time() + 3600),  # 1 hour
            resource=authorization_code.resource,
        )
        self._access_tokens[access_token_str] = access_token

        # Generate refresh token
        refresh_token_str = secrets.token_urlsafe(32)
        refresh_token = RefreshToken(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time() + 86400 * 30),  # 30 days
        )
        self._refresh_tokens[refresh_token_str] = refresh_token

        # Delete the authorization code (one-time use)
        if authorization_code.code in self._authorization_codes:
            del self._authorization_codes[authorization_code.code]

        logger.info(f"Issued access token (expires in 1h) and refresh token (expires in 30d)")

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token_str,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str
    ) -> Optional[RefreshToken]:
        """
        Load a refresh token.

        Args:
            client: The client requesting the refresh token
            refresh_token: The refresh token string

        Returns:
            RefreshToken object, or None if not found/expired
        """
        token_obj = self._refresh_tokens.get(refresh_token)

        if not token_obj:
            logger.warning(f"Refresh token not found")
            return None

        # Verify it belongs to this client
        if token_obj.client_id != client.client_id:
            logger.warning(f"Refresh token client mismatch")
            return None

        # Check expiration
        if token_obj.expires_at and token_obj.expires_at < time.time():
            logger.warning(f"Refresh token expired")
            del self._refresh_tokens[refresh_token]
            return None

        return token_obj

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str]
    ) -> OAuthToken:
        """
        Exchange refresh token for new access token.

        Args:
            client: The client exchanging the refresh token
            refresh_token: The refresh token to exchange
            scopes: Requested scopes

        Returns:
            New OAuth token with access_token

        Raises:
            TokenError: If the exchange fails
        """
        logger.info(f"Exchanging refresh token for new access token (client={client.client_id[:8]}...)")

        # Validate scopes (must be subset of original scopes)
        if scopes and not all(scope in refresh_token.scopes for scope in scopes):
            raise TokenError(
                error="invalid_scope",
                error_description="Requested scopes exceed original grant"
            )

        # Generate new access token
        access_token_str = secrets.token_urlsafe(32)
        access_token = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=scopes or refresh_token.scopes,
            expires_at=int(time.time() + 3600),  # 1 hour
        )
        self._access_tokens[access_token_str] = access_token

        logger.info(f"Issued new access token (expires in 1h)")

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(access_token.scopes) if access_token.scopes else None,
        )

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        """
        Load and verify an access token (required by OAuthAuthorizationServerProvider).

        This method is called by FastMCP's ProviderTokenVerifier wrapper to validate
        Bearer tokens for MCP requests.

        Args:
            token: The access token to verify

        Returns:
            AccessToken object if valid, None otherwise
        """
        logger.info(f"load_access_token called with token: {token[:16]}...")

        # Check for pre-allowed token from environment
        pre_allowed_token = os.getenv("OAUTH_PRE_ALLOWED_TOKEN")
        if pre_allowed_token and token == pre_allowed_token:
            logger.info(f"Token matched pre-allowed token from environment")
            # Return a non-expiring access token for the pre-allowed token
            return AccessToken(
                token=token,
                client_id=self.client_id,
                scopes=[],
                expires_at=None,  # Non-expiring
                resource=None,
            )

        # Fall back to OAuth flow tokens
        access_token = self._access_tokens.get(token)

        if not access_token:
            logger.warning(f"Token not found in storage: {token[:16]}...")
            logger.debug(f"Currently stored tokens: {list(self._access_tokens.keys())}")
            return None

        # Check expiration
        if access_token.expires_at and access_token.expires_at < time.time():
            logger.warning(f"Access token expired")
            del self._access_tokens[token]
            return None

        logger.info(f"Token verified successfully for client {access_token.client_id[:8]}...")
        return access_token


def get_oauth_config() -> Optional[tuple[str, str, str]]:
    """
    Load OAuth configuration from environment variables.

    Returns:
        Tuple of (client_id, client_secret, issuer_url) if configured, None otherwise
    """
    client_id = os.getenv("OAUTH_CLIENT_ID")
    client_secret = os.getenv("OAUTH_CLIENT_SECRET")
    issuer_url = os.getenv("OAUTH_ISSUER_URL")

    if not client_id or not client_secret:
        return None

    # Validate they're not empty
    if not client_id.strip() or not client_secret.strip():
        logger.warning("OAUTH_CLIENT_ID or OAUTH_CLIENT_SECRET is empty")
        return None

    # Default issuer URL if not provided
    if not issuer_url:
        # Try to infer from host/port
        host = os.getenv("THINGS_FASTMCP_HOST", "127.0.0.1")
        port = os.getenv("THINGS_FASTMCP_PORT", "8009")

        # Use http for localhost, https for others (though localhost is recommended for testing)
        if host in ("127.0.0.1", "localhost"):
            issuer_url = f"http://{host}:{port}"
        else:
            issuer_url = f"https://{host}:{port}"
            logger.warning(
                "Using HTTPS for issuer URL. Ensure you have TLS configured or use localhost for testing."
            )

    return (client_id.strip(), client_secret.strip(), issuer_url.strip())


def create_oauth_provider() -> Optional[SimpleOAuthProvider]:
    """
    Create an OAuth provider if configuration is available.

    Returns:
        SimpleOAuthProvider instance if configured, None otherwise
    """
    config = get_oauth_config()
    if not config:
        logger.info("OAuth not configured - authentication disabled")
        return None

    client_id, client_secret, issuer_url = config
    provider = SimpleOAuthProvider(client_id, client_secret, issuer_url)
    logger.info(f"OAuth provider created with issuer: {issuer_url}")

    return provider
