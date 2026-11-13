package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * A function-calling model over plain HTTP: builds one request, sends it with bounded retries and
 * decodes the response into a {@link Proposal}.
 *
 * <p>Keys come from builder options or environment variables and never appear in errors,
 * {@link #toString()} or results. Subclasses define {@link #url()}, {@link #payload},
 * {@link #decode} and usually {@link #authHeaders()} and {@link #usage}.
 */
public abstract class Provider implements Generator {
    /** Why an output holds no complete proposal. Such a proposal is rejected and repaired. */
    public static final Map<String, String> INCOMPLETE = incompleteReasons();

    private static final Pattern SAFE_ID = Pattern.compile("[A-Za-z0-9._:-]{1,128}");
    private static final Pattern SAFE_CODE = Pattern.compile("[A-Za-z0-9_.-]{1,64}");

    private static Map<String, String> incompleteReasons() {
        Map<String, String> reasons = new LinkedHashMap<>();
        reasons.put("length", "The response reached the output token limit before a complete proposal.");
        reasons.put("content_filter", "The provider's content filter stopped the response before a complete proposal.");
        reasons.put("refusal", "The response was a refusal and contains no proposal.");
        reasons.put("prompt_filter", "The provider's content filter rejected the request; no response was generated.");
        reasons.put("malformed_call", "The model produced a malformed function call.");
        reasons.put("abnormal_stop", "The response stopped abnormally before a complete proposal.");
        return Collections.unmodifiableMap(reasons);
    }

    /**
     * What distinguishes one provider class: its name in results, default endpoint and the
     * environment variables it reads.
     */
    protected record Kind(String name, String defaultBaseUrl, List<String> baseUrlEnv, List<String> keyEnv,
                          boolean needsKey) {
    }

    /** A decoded response: calls ({@code null} if unreadable), parse errors, text and the reason. */
    protected record Decoded(List<Call> calls, List<String> parseErrors, String content, String incomplete) {
        public static Decoded incomplete(String reason) {
            return new Decoded(null, List.of(INCOMPLETE.get(reason)), null, reason);
        }

        /** A complete proposal. A response with neither a call nor text is a decision to make no call. */
        public static Decoded complete(List<Call> calls, List<String> errors, List<String> text) {
            return new Decoded(calls, errors, text.isEmpty() ? null : String.join("\n", text), null);
        }
    }

    private final Kind kind;
    protected final String model;
    private final String apiKey;
    protected final String baseUrl;
    protected final Integer maxOutputTokens;
    protected final Number temperature;
    protected final String systemPrompt;
    protected final double timeoutSeconds;
    protected final int maxRetries;
    private final Map<String, String> extraHeaders;
    private final HttpTransport transport;
    private final boolean ownsTransport;
    private final Sleeper sleeper;

    /**
     * @param b       the options
     * @param kind    the provider's identity and environment variables
     * @param model   the model name to send
     * @param baseUrl an explicit endpoint (Azure's), or {@code null} for the builder's, the
     *                environment's or the default
     */
    protected Provider(Builder<?, ?> b, Kind kind, String model, String baseUrl) {
        String cls = getClass().getSimpleName();
        this.kind = kind;
        if (model == null || Py.strip(model).isEmpty()) {
            throw new IllegalArgumentException(cls + " needs a model name.");
        }
        if (b.maxRetries < 0) {
            throw new IllegalArgumentException("maxRetries must be a nonnegative integer.");
        }
        if (b.maxOutputTokens != null && b.maxOutputTokens < 1) {
            throw new IllegalArgumentException("maxOutputTokens must be a positive integer or null.");
        }
        if (Double.isNaN(b.timeoutSeconds) || Double.isInfinite(b.timeoutSeconds) || b.timeoutSeconds <= 0) {
            throw new IllegalArgumentException("timeout must be a positive number of seconds.");
        }
        if (b.temperature != null && (Double.isNaN(b.temperature.doubleValue())
                || Double.isInfinite(b.temperature.doubleValue()))) {
            throw new IllegalArgumentException("temperature must be a finite number or null.");
        }
        if (b.systemPrompt == null || Py.strip(b.systemPrompt).isEmpty()) {
            throw new IllegalArgumentException("systemPrompt must be nonempty text.");
        }
        this.model = model;
        String key = b.apiKey != null && !b.apiKey.isEmpty() ? b.apiKey : Env.first(kind.keyEnv());
        if (kind.needsKey() && key == null) {
            String names = kind.keyEnv().isEmpty() ? "apiKey" : String.join(" or ", kind.keyEnv());
            throw new IllegalArgumentException(cls + " needs an API key: pass apiKey(...) or set " + names + ".");
        }
        if (key != null && (key.contains("\n") || key.contains("\r"))) {
            throw new IllegalArgumentException("API keys must be single-line strings.");
        }
        this.apiKey = key;
        String candidate = baseUrl != null ? baseUrl : firstNonEmpty(b.baseUrl, Env.first(kind.baseUrlEnv()),
                kind.defaultBaseUrl());
        this.baseUrl = Py.baseUrl(candidate);
        if (this.baseUrl == null) {
            throw new IllegalArgumentException(cls + " needs an http(s) baseUrl without credentials, query or fragment.");
        }
        this.maxOutputTokens = b.maxOutputTokens;
        this.temperature = b.temperature;
        this.systemPrompt = b.systemPrompt;
        this.timeoutSeconds = b.timeoutSeconds;
        this.maxRetries = b.maxRetries;
        this.extraHeaders = new LinkedHashMap<>(b.headers);
        this.transport = b.transport != null ? b.transport
                : new JdkHttpTransport(Duration.ofMillis(Math.round(b.timeoutSeconds * 1000)), true);
        this.ownsTransport = b.transport == null;
        this.sleeper = b.sleeper == null ? Sleeper.SYSTEM : b.sleeper;
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }
        return null;
    }

    // Subclass interface ---------------------------------------------------------------
    /** The request URL. */
    public abstract String url();

    /** The authentication headers; {@code Authorization: Bearer <key>} by default. */
    protected Map<String, String> authHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        if (apiKey != null) {
            headers.put("Authorization", "Bearer " + apiKey);
        }
        return headers;
    }

    /** The request body. */
    protected abstract ObjectNode payload(ArrayNode messages, ArrayNode tools, ArrayNode history);

    /** The proposal in a successful response body. */
    protected abstract Decoded decode(ObjectNode data);

    /** The token counts in a successful response body. */
    protected Map<String, Long> usage(ObjectNode data) {
        return new LinkedHashMap<>();
    }

    /** Whether an error response means the provider filtered the prompt (an unreadable candidate). */
    protected boolean filteredRequest(int statusCode, String code) {
        return false;
    }

    /** The API key, for {@link #authHeaders()} overrides. */
    protected final String apiKey() {
        return apiKey;
    }

    // Shared machinery ------------------------------------------------------------------
    /** The provider's name in proposals, for example {@code openai}. */
    public String name() {
        return kind.name();
    }

    public String model() {
        return model;
    }

    public String baseUrl() {
        return baseUrl;
    }

    /** The request headers: extra headers, then authentication, then the content type. */
    protected Map<String, String> headers() {
        Map<String, String> headers = new LinkedHashMap<>(extraHeaders);
        headers.putAll(authHeaders());
        headers.put("Content-Type", "application/json");
        return headers;
    }

    @Override
    public void close() {
        if (ownsTransport) {
            transport.close();
        }
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(model=" + Py.strRepr(model) + ", baseUrl=" + Py.strRepr(baseUrl) + ")";
    }

    @Override
    public Proposal generate(ArrayNode messages, ArrayNode tools, ArrayNode history) {
        ObjectNode payload = payload(messages, tools == null ? Json.array() : tools, history);
        byte[] body = Json.compact(payload).getBytes(StandardCharsets.UTF_8);
        long start = System.nanoTime();
        double waited = 0.0;
        int attempts = 0;
        HttpTransport.Response response = null;
        Duration timeout = Duration.ofMillis(Math.round(timeoutSeconds * 1000));
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            attempts++;
            try {
                response = transport.send(new HttpTransport.Request("POST", url(), headers(), body, timeout));
            } catch (IOException e) {
                if (attempt == maxRetries) {
                    throw new ProviderException(name() + " transport failed after " + attempts + " attempts ("
                            + e.getClass().getSimpleName() + ").", null, null, attempts, null);
                }
                waited += pause(Math.min(Math.pow(2, attempt), 8));
                continue;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new ProviderException(name() + " request was interrupted.", null, null, attempts, null);
            }
            int status = response.status();
            if ((status == 408 || status == 429 || status >= 500) && attempt < maxRetries) {
                waited += pause(retryDelay(response, attempt));
                continue;
            }
            break;
        }
        double latency = Math.max(0.0, (System.nanoTime() - start) / 1e9 - waited);
        ObjectNode data = Json.object();
        Map<String, Long> usage = new LinkedHashMap<>();
        Decoded decoded;
        if (!response.isSuccess()) {
            String code = errorCode(response);
            if (!filteredRequest(response.status(), code)) {
                String detail = code != null ? " (" + code + ")" : "";
                throw new ProviderException(name() + " HTTP " + response.status() + detail
                        + "; check the model, key and endpoint.", response.status(), requestId(response), attempts,
                        code);
            }
            decoded = Decoded.incomplete("prompt_filter");
        } else {
            try {
                JsonNode parsed = Json.parse(response.body());
                if (!parsed.isObject()) {
                    throw new ProviderException("The response was not a JSON object.");
                }
                data = (ObjectNode) parsed;
                decoded = decode(data);
                usage = usage(data);
            } catch (ProviderException e) {
                throw e.withContext(response.status(), requestId(response), attempts);
            } catch (Py.ProtocolMismatch | IllegalArgumentException | ClassCastException | NullPointerException
                     | IndexOutOfBoundsException | UnsupportedOperationException e) {
                throw new ProviderException("The " + name() + " response did not match its protocol.",
                        response.status(), requestId(response), attempts, null);
            }
        }
        JsonNode resolved = Py.truthy(data.get("model")) ? data.get("model") : data.get("modelVersion");
        return Proposal.builder()
                .calls(decoded.calls())
                .content(decoded.content())
                .parseErrors(decoded.parseErrors())
                .incomplete(decoded.incomplete())
                .provider(name())
                .requestedModel(model)
                .model(Py.text(resolved))
                .usage(usage)
                .latencySeconds(latency)
                .retryWaitSeconds(waited)
                .apiCalls(attempts)
                .build();
    }

    private double pause(double seconds) {
        long started = System.nanoTime();
        try {
            sleeper.sleep(seconds);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ProviderException(name() + " was interrupted while waiting to retry.");
        }
        return (System.nanoTime() - started) / 1e9;
    }

    static double retryDelay(HttpTransport.Response response, int attempt) {
        double fallback = Math.pow(2, attempt);
        String header = response.header("retry-after");
        double delay = fallback;
        if (header != null) {
            Double parsed = Py.parseFloat(header);
            delay = parsed == null ? fallback : parsed;
        }
        if (Double.isNaN(delay) || Double.isInfinite(delay)) {
            return Math.min(fallback, 30.0);
        }
        return Math.max(0.0, Math.min(delay, 30.0));
    }

    private String requestId(HttpTransport.Response response) {
        for (String header : new String[]{"x-request-id", "request-id", "apim-request-id", "x-goog-request-id"}) {
            String value = response.header(header);
            if (value != null && SAFE_ID.matcher(value).matches() && !(apiKey != null && value.contains(apiKey))) {
                return value;
            }
        }
        return null;
    }

    private static String errorCode(HttpTransport.Response response) {
        JsonNode body;
        try {
            body = Json.parse(response.body());
        } catch (IllegalArgumentException e) {
            return null;
        }
        if (!body.isObject()) {
            return null;
        }
        JsonNode error = body.get("error");
        JsonNode code = null;
        if (error != null && error.isObject()) {
            JsonNode value = error.get("code");
            if (value != null && value.isTextual()) {
                code = value;
            } else {
                code = Py.truthy(error.get("type")) ? error.get("type") : error.get("status");
            }
        }
        String text = Py.text(code);
        return text != null && SAFE_CODE.matcher(text).matches() ? text : null;
    }

    // Decoding helpers ---------------------------------------------------------------------
    /**
     * Appends one call. Malformed arguments stay a rejected, repairable call, never a no call.
     *
     * <p>Chat Completions sends the arguments as JSON text ({@code encoded}). Some compatible servers
     * send the object itself, or null for no arguments; both are read like the other protocols.
     */
    protected static void appendCall(List<Call> calls, List<String> errors, JsonNode name, JsonNode arguments,
                                     boolean encoded) {
        String text = Py.text(name);
        if (text == null || text.isEmpty()) {
            throw new ProviderException("The response contained an invalid function name.");
        }
        JsonNode value = arguments;
        String error = null;
        if (encoded && value != null && value.isTextual()) {
            String raw = value.textValue();
            if (Py.strip(raw).isEmpty()) {
                value = Json.object();
            } else {
                try {
                    value = Json.parse(raw);
                } catch (IllegalArgumentException e) {
                    value = null;
                    error = "Function arguments are not valid JSON.";
                }
            }
        }
        if (error == null) {
            if (Py.isNone(value)) {
                value = Json.object();
            }
            if (!value.isObject()) {
                value = null;
                error = "Function arguments must be a JSON object.";
            } else if (!Json.isFinite(value)) {
                value = null;
                error = "Function arguments must contain finite JSON values.";
            }
        }
        calls.add(new Call(text, value));
        if (error != null) {
            errors.add(error);
        }
    }

    /** A token count: absent or null gives {@code null}; anything but a nonnegative integer is invalid. */
    protected static Long count(JsonNode value) {
        if (Py.isNone(value)) {
            return null;
        }
        if (!value.isIntegralNumber() || value.bigIntegerValue().signum() < 0 || !value.canConvertToLong()) {
            throw new ProviderException("The response contained invalid token usage.");
        }
        return value.longValue();
    }

    /** The counts that are known. */
    protected static Map<String, Long> withoutNulls(Map<String, Long> counts) {
        Map<String, Long> result = new LinkedHashMap<>();
        counts.forEach((key, value) -> {
            if (value != null) {
                result.put(key, value);
            }
        });
        return result;
    }

    /**
     * Options shared by every provider. Subclass builders add their own.
     *
     * @param <B> the builder type
     * @param <P> the provider it builds
     */
    public abstract static class Builder<B extends Builder<B, P>, P extends Provider> {
        protected final String model;
        protected String apiKey;
        protected String baseUrl;
        protected Integer maxOutputTokens = 4096;
        protected Number temperature;
        protected String systemPrompt = Prompts.SYSTEM_PROMPT;
        protected double timeoutSeconds = 120.0;
        protected int maxRetries = 3;
        protected final Map<String, String> headers = new LinkedHashMap<>();
        protected HttpTransport transport;
        protected Sleeper sleeper;

        protected Builder(String model) {
            this.model = model;
        }

        protected abstract B self();

        /** Builds the provider; invalid options raise {@link IllegalArgumentException}. */
        public abstract P build();

        /** The API key; by default it comes from the provider's environment variable. */
        public B apiKey(String apiKey) {
            this.apiKey = apiKey;
            return self();
        }

        /** The API base URL (for example {@code https://api.openai.com/v1}). */
        public B baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return self();
        }

        /** The output token limit (4096 by default); {@code null} sends none where the API allows. */
        public B maxOutputTokens(Integer maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return self();
        }

        /** The sampling temperature; none is sent by default. */
        public B temperature(Number temperature) {
            this.temperature = temperature;
            return self();
        }

        /** Replaces the generator system prompt (the paper's {@link Prompts#SYSTEM_PROMPT} by default). */
        public B systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return self();
        }

        /** The request timeout (120 seconds by default). */
        public B timeout(Duration timeout) {
            this.timeoutSeconds = timeout.toMillis() / 1000.0;
            return self();
        }

        /** HTTP retries after the first attempt for 408, 429 and 5xx responses (3 by default). */
        public B maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return self();
        }

        /** An extra request header. */
        public B header(String name, String value) {
            this.headers.put(name, value);
            return self();
        }

        /** The HTTP transport; the caller then owns it. */
        public B transport(HttpTransport transport) {
            this.transport = transport;
            return self();
        }

        /** How to wait between retries. */
        public B sleeper(Sleeper sleeper) {
            this.sleeper = sleeper;
            return self();
        }
    }
}
