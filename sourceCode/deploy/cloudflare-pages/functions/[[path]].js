// Cloudflare Pages Function: reverse proxy mọi request của *.pages.dev về origin (Nginx trên server).
// ORIGIN_URL và ORIGIN_SECRET được cấu hình dạng secret trong Pages project (deploy.sh pages).

const HOP_BY_HOP_HEADERS = ["connection", "keep-alive", "transfer-encoding", "upgrade", "host"];

export async function onRequest({ request, env }) {
  if (!env.ORIGIN_URL || !env.ORIGIN_SECRET) {
    return new Response("Proxy chưa được cấu hình", { status: 500 });
  }
  const incoming = new URL(request.url);
  const origin = new URL(env.ORIGIN_URL);
  const target = new URL(incoming.pathname + incoming.search, origin);

  const headers = new Headers(request.headers);
  HOP_BY_HOP_HEADERS.forEach((name) => headers.delete(name));
  headers.set("X-Origin-Auth", env.ORIGIN_SECRET);
  headers.set("X-Forwarded-Host", incoming.host);

  const hasBody = !["GET", "HEAD"].includes(request.method);
  let response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      redirect: "manual",
    });
  } catch (error) {
    return new Response(`Origin không phản hồi: ${error.message}`, { status: 502 });
  }

  // Redirect tuyệt đối trỏ về origin phải đổi lại thành domain công khai
  const result = new Response(response.body, response);
  const location = result.headers.get("Location");
  if (location) {
    const redirectUrl = new URL(location, target);
    if (redirectUrl.host === origin.host) {
      redirectUrl.protocol = "https:";
      redirectUrl.host = incoming.host;
      result.headers.set("Location", redirectUrl.toString());
    }
  }
  return result;
}
