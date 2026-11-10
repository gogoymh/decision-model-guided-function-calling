/** HTTP over an injectable `fetch`, environment variables, and URL checks shared by every client. */

/** The part of a `fetch` response DM-Call reads. The global `Response` satisfies it. */
export interface ResponseLike {
  readonly status: number;
  readonly headers: { get(name: string): string | null };
  text(): Promise<string>;
}

/** The request DM-Call makes. */
export interface RequestInitLike {
  method: string;
  headers: Record<string, string>;
  body?: string;
  signal?: AbortSignal;
  /** Redirects never forward provider credentials to another endpoint. */
  redirect: "manual";
}

/** A `fetch` implementation. The global `fetch` (Node 18+) is the default; pass your own to test or proxy. */
export type FetchLike = (url: string, init: RequestInitLike) => Promise<ResponseLike>;

/** Waits `ms` milliseconds; injectable so tests do not wait. */
export type Sleep = (ms: number) => Promise<void>;

export const defaultSleep: Sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export interface HttpResponse {
  status: number;
  headers: { get(name: string): string | null };
  text: string;
}

export function resolveFetch(fetchImpl: FetchLike | undefined): FetchLike {
  if (fetchImpl !== undefined) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch must be a function.");
    return fetchImpl;
  }
  return (url, init) => {
    const global = (globalThis as { fetch?: FetchLike }).fetch;
    if (typeof global !== "function") {
      return Promise.reject(new Error("No fetch implementation is available; pass fetch in the options."));
    }
    return global(url, init);
  };
}

function timeoutSignal(ms: number): AbortSignal | undefined {
  const timeout = (AbortSignal as unknown as { timeout?: (ms: number) => AbortSignal }).timeout;
  return typeof timeout === "function" ? timeout.call(AbortSignal, ms) : undefined;
}

/** One request, body read included. Any failure to get a response throws (a transport error). */
export async function send(fetchImpl: FetchLike, url: string, method: string, headers: Record<string, string>,
  body: string | undefined, timeoutMs: number): Promise<HttpResponse> {
  const init: RequestInitLike = { method, headers, redirect: "manual" };
  if (body !== undefined) init.body = body;
  const signal = timeoutSignal(timeoutMs);
  if (signal) init.signal = signal;
  const response = await fetchImpl(url, init);
  const text = await response.text();
  return { status: response.status, headers: response.headers, text };
}

export function now(): number {
  return performance.now();
}

/** Sleeps and reports the milliseconds actually waited. */
export async function wait(sleep: Sleep, ms: number): Promise<number> {
  const started = now();
  await sleep(ms);
  return now() - started;
}

export function env(name: string): string | undefined {
  const variables = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
  return variables?.[name];
}

/** The first nonempty environment variable among `names`, stripped. */
export function firstEnv(names: readonly string[]): string | undefined {
  for (const name of names) {
    const value = env(name)?.trim();
    if (value) return value;
  }
  return undefined;
}

/** Python's `float(text)`; `undefined` where Python raises ValueError. */
export function pythonFloat(text: string): number | undefined {
  const value = text.trim().toLowerCase().replace(/^\+/, "");
  const sign = value.startsWith("-") ? -1 : 1;
  const body = value.replace(/^-/, "");
  if (/^(inf|infinity)$/.test(body)) return sign * Infinity;
  if (body === "nan") return NaN;
  if (!/^(\d(_?\d)*)?(\.(\d(_?\d)*)?)?(e[+-]?\d(_?\d)*)?$/.test(body) || !/\d/.test(body.split("e")[0]!)) return undefined;
  return sign * Number(body.replace(/_/g, ""));
}

export interface UrlParts {
  scheme: string;
  netloc: string;
  path: string;
  query: string | null;
  fragment: string | null;
  hostname: string | null;
}

/** Python's `urllib.parse.urlsplit`, the parts the clients check. */
export function urlsplit(url: string): UrlParts {
  let rest = url;
  let scheme = "";
  const schemeMatch = /^([A-Za-z][A-Za-z0-9+.-]*):/.exec(rest);
  if (schemeMatch) {
    scheme = schemeMatch[1]!.toLowerCase();
    rest = rest.slice(schemeMatch[0].length);
  }
  let netloc = "";
  if (rest.startsWith("//")) {
    const end = rest.slice(2).search(/[/?#]/);
    netloc = end === -1 ? rest.slice(2) : rest.slice(2, 2 + end);
    rest = end === -1 ? "" : rest.slice(2 + end);
  }
  let fragment: string | null = null;
  const hash = rest.indexOf("#");
  if (hash !== -1) {
    fragment = rest.slice(hash + 1);
    rest = rest.slice(0, hash);
  }
  let query: string | null = null;
  const question = rest.indexOf("?");
  if (question !== -1) {
    query = rest.slice(question + 1);
    rest = rest.slice(0, question);
  }
  const host = netloc.includes("@") ? netloc.slice(netloc.lastIndexOf("@") + 1) : netloc;
  let hostname: string | null;
  if (host.startsWith("[")) hostname = host.slice(1, host.indexOf("]") === -1 ? undefined : host.indexOf("]")).toLowerCase();
  else hostname = (host.includes(":") ? host.slice(0, host.indexOf(":")) : host).toLowerCase();
  if (hostname === "") hostname = null;
  return { scheme, netloc, path: rest, query, fragment, hostname };
}

/** A base URL with http(s), a host, no credentials, query or fragment, and no trailing slash; else `null`. */
export function checkBaseUrl(value: string | null | undefined): string | null {
  const url = (value ?? "").trim().replace(/\/+$/, "");
  const parts = urlsplit(url);
  if ((parts.scheme !== "http" && parts.scheme !== "https") || !parts.netloc || parts.netloc.includes("@")
    || parts.query || parts.fragment) return null;
  return url;
}

/** Python's `urllib.parse.quote(text, safe="")`. */
export function quote(text: string): string {
  return encodeURIComponent(text).replace(/[!'()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}

/** A diagnostic request ID from response headers, never one that contains a secret. */
export function requestId(headers: { get(name: string): string | null }, names: readonly string[],
  secrets: readonly (string | undefined)[]): string | null {
  for (const name of names) {
    const value = headers.get(name) ?? "";
    if (/^[A-Za-z0-9._:-]{1,128}$/.test(value) && !secrets.some((secret) => secret && value.includes(secret))) {
      return value;
    }
  }
  return null;
}

export function errorName(error: unknown): string {
  if (error instanceof Error && /^[A-Za-z0-9_]{1,64}$/.test(error.name)) return error.name;
  return "Error";
}
