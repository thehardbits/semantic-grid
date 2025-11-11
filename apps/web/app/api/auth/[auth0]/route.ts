export const runtime = "nodejs";


import type { AppRouteHandlerFnContext } from "@auth0/nextjs-auth0";
import {
  handleAuth,
  handleCallback,
  handleLogin,
  handleLogout,
} from "@auth0/nextjs-auth0";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

function getUrls(req: NextRequest) {
  const host = (req.headers as any).get("host");
  const protocol = process.env.NODE_ENV === "development" ? "http" : "https";
  const redirectUri = `${protocol}://${host}/api/auth/callback`;
  // get return to url from query params
  const returnTo =
    req.nextUrl.searchParams.get("returnTo") || `${protocol}://${host}`;
  return {
    redirectUri,
    returnTo,
  };
}

const GET = handleAuth({
  onError(req: NextRequest, error: any) {
    console.log(
      "auth error",
      error.status,
      error.cause.error,
      error.cause.errorDescription,
    );
    const { returnTo } = getUrls(req);
    if (
      error.status === 400
      // && error.cause.error === "access_denied"
    ) {
      return NextResponse.redirect(`${returnTo}`);
    }
    return NextResponse.error();
  },

  callback(req: NextRequest, ctx: AppRouteHandlerFnContext) {
    const { redirectUri } = getUrls(req);
    try {
      // console.log("callback", req.nextUrl.searchParams);
      // console.log("params", req, ctx, redirectUri);
      // console.log("params", redirectUri);
      let result =  handleCallback(req, ctx, {
        redirectUri,
      });
      // console.log("callback result", result);
      return result;
    } catch (e: any) {
      console.log("callback error", e);
      return NextResponse.error();
    }
  },

  login(req: NextRequest, ctx: AppRouteHandlerFnContext) {
    const { returnTo, redirectUri } = getUrls(req);
    return handleLogin(req, ctx, {
      returnTo,
      authorizationParams: {
        // prompt: "none",
        redirectUri,
        scope:
          "openid profile email create:session update:session create:request update:request admin:requests admin:sessions",
        audience: process.env.AUTH0_AUDIENCE,
      },
    });
  },

  logout(req: NextRequest, ctx: AppRouteHandlerFnContext) {
    const { returnTo } = getUrls(req);
    return handleLogout(req, ctx, {
      returnTo,
    });
  },
});

export { GET };
