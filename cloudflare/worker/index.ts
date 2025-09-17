export default {
    async fetch(req: Request, env: Env): Promise<Response> {
      const url = new URL(req.url);
      if (!["/v1/embed", "/healthz"].includes(url.pathname)) {
        return new Response("Not found", { status: 404 });
      }
      const target = env.RUNPOD_URL + url.pathname + url.search;
  
      const h = new Headers(req.headers);
      h.set("x-internal-key", env.INTERNAL_KEY);
  
      const len = Number(req.headers.get("content-length") || "0");
      if (len > 2_000_000) return new Response("Payload too large", { status: 413 });
  
      const res = await fetch(target, { method: req.method, headers: h, body: req.body });
      return new Response(res.body, { status: res.status, headers: res.headers });
    }
  } satisfies ExportedHandler<{ RUNPOD_URL: string; INTERNAL_KEY: string }>;