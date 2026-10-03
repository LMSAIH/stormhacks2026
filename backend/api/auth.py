from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from config import FRONTEND_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET


router = APIRouter(prefix="/api/auth", tags=["authentication"])
oauth = OAuth()
oauth.register(
	name="google",
	client_id=GOOGLE_CLIENT_ID,
	client_secret=GOOGLE_CLIENT_SECRET,
	server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
	client_kwargs={"scope": "openid email profile"},
)


def require_google_credentials() -> None:
	if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
		raise HTTPException(status_code=503, detail="Google OAuth is not configured")


@router.get("/google/login", name="google_login")
async def google_login(request: Request):
	require_google_credentials()
	callback_url = request.url_for("google_callback")
	return await oauth.google.authorize_redirect(request, callback_url)


@router.get("/google/callback", name="google_callback")
async def google_callback(request: Request) -> dict:
	require_google_credentials()
	try:
		token = await oauth.google.authorize_access_token(request)
	except Exception as error:
		raise HTTPException(status_code=401, detail="Google sign-in failed") from error

	userinfo = token.get("userinfo") or {}
	if not userinfo.get("sub") or not userinfo.get("email_verified"):
		raise HTTPException(status_code=401, detail="Google account identity was not verified")

	user = {
		"id": userinfo["sub"],
		"email": userinfo.get("email"),
		"name": userinfo.get("name"),
		"picture": userinfo.get("picture"),
	}
	request.session["user"] = user
	return RedirectResponse(FRONTEND_URL, status_code=303)


@router.get("/me")
async def current_user(request: Request) -> dict:
	user = await require_authenticated_user(request)
	return {"user": user}


async def require_authenticated_user(request: Request) -> dict:
	user = get_authenticated_user(request)
	if user is None:
		raise HTTPException(status_code=401, detail="Sign-in required")
	return user


def get_authenticated_user(request: Request) -> dict | None:
	return request.session.get("user")


@router.post("/logout")
async def logout(request: Request) -> dict:
	request.session.clear()
	return {"logged_out": True}