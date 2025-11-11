from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from fm_app.config import get_settings

import logging
LOGGER = logging.getLogger(__name__)

class UnauthorizedException(HTTPException):
    def __init__(self, detail: str, **kwargs):
        """Returns HTTP 403"""
        super().__init__(status.HTTP_403_FORBIDDEN, detail=detail)


class UnauthenticatedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Requires authentication"
        )


class VerifyToken:
    """Does all the token verification using PyJWT"""

    def __init__(self):
        self.config = get_settings()

        # This gets the JWKS from a given URL and does processing so you can
        # use any of the keys available
        jwks_url = f"https://{self.config.auth0_domain}/.well-known/jwks.json"
        # LOGGER.info(f"JWKS_UTL {jwks_url} ");
        self.jwks_client = jwt.PyJWKClient(jwks_url)

    async def verify(
        self,
        security_scopes: SecurityScopes,
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer()),
    ):
        # LOGGER.info(f"verify user token {token}")
        # LOGGER.info(f"verify user jwks_client {self.jwks_client}")
        if token is None:
            return None
            # raise UnauthenticatedException

        # This gets the 'kid' from the passed token
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(
                token.credentials
            ).key
        except jwt.exceptions.PyJWKClientError:
            LOGGER.error("PyJWKClientError error")
            return None
            # raise UnauthorizedException(str(error))
        except jwt.exceptions.DecodeError as error:
            LOGGER.error(f"Decode error {str(error)}")
            raise UnauthorizedException(str(error))

        try:
            unverified_claims = jwt.decode(
                    token.credentials,
                    options={"verify_signature": False, "verify_aud": False}  # don't verify yet
            )
            LOGGER.info("iss=%s aud=%s sub=%s", 
                unverified_claims.get("iss"),
                unverified_claims.get("aud"),
                unverified_claims.get("sub"))
            payload = jwt.decode(
                token.credentials,
                signing_key,
                algorithms=self.config.auth0_algorithms,
                audience=self.config.auth0_api_audience,
                issuer=self.config.auth0_issuer,
            )
        except Exception as error:
            # print(error, token.credentials, signing_key)
            LOGGER.error(f"Payload decode error {str(error)}")
            return None
            # raise UnauthorizedException(str(error))

        # print("payload", payload)
        token_scopes = payload.get("permissions", [])

        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                print("scope not in token scopes", scope, token_scopes)
                return None
                # raise HTTPException(
                #    status_code=403,
                #    detail=f"Missing required scope: {scope}",
                # )

        return payload


class VerifyGuestToken:
    """Does all the token verification using PyJWT"""

    def __init__(self):
        self.config = get_settings()

        # This gets the JWKS from a given URL and does processing so you can
        # use any of the keys available
        jwks_url = f"{self.config.guest_auth_host}/.well-known/jwks.json"
        self.jwks_client = jwt.PyJWKClient(jwks_url)

    async def verify(
        self,
        security_scopes: SecurityScopes,
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer()),
    ):
        # print("verify guest token", token)
        if token is None:
            return None
            # raise UnauthenticatedException

        # This gets the 'kid' from the passed token
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(
                token.credentials
            ).key
            # print("verify sign key", signing_key)

        except jwt.exceptions.PyJWKClientError:
            # print("verify error", error)
            # raise UnauthorizedException(str(error))
            return None
        except jwt.exceptions.DecodeError:
            # print("verify error", error)
            # raise UnauthorizedException(str(error))
            return None

        try:
            payload = jwt.decode(
                token.credentials,
                signing_key,
                algorithms=self.config.auth0_algorithms,
                audience=self.config.auth0_api_audience,
                issuer=self.config.guest_auth_issuer,
            )
            # print("guest payload", payload)

        except Exception:
            # print("decode error", error)
            # raise UnauthorizedException(str(error))
            return None

        return payload
