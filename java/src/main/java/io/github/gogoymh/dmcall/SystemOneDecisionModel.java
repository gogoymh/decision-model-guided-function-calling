package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * A decision model reached over the {@code POST {base_url}/v1/systemone} protocol, which Jev and the
 * PPLX Decider both speak.
 *
 * <p>The threshold is the pass mark of both questions (0.5 in the paper); a function or argument
 * threshold overrides one of them. {@code maxRetries} counts HTTP retries after the first attempt
 * (on 408, 429, 500, 502, 503, 504 and 529) and has nothing to do with DM-Call's repair limit. To
 * connect a decision model that speaks another protocol, subclass this class and override
 * {@link #ask(ObjectNode, ObjectNode)}.
 */
public class SystemOneDecisionModel implements DecisionModel {
    /** HTTP statuses that are retried. */
    public static final Set<Integer> RETRY_STATUSES = Set.of(408, 429, 500, 502, 503, 504, 529);
    private static final String[] OBSERVABLE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"};
    private static final Pattern SAFE_ID = Pattern.compile("[A-Za-z0-9._:-]{1,128}");

    private final double functionThreshold;
    private final double argumentsThreshold;
    private final String baseUrl;
    private final String model;
    private final double timeoutSeconds;
    private final int maxRetries;
    private final Map<String, String> headers;
    private final HttpTransport transport;
    private final boolean ownsTransport;
    private final Sleeper sleeper;

    /** The answers to one {@link #ask} request. */
    public record Answer(Map<String, Double> probabilities, String model, Map<String, Long> usage, int apiCalls,
                         double retryWaitSeconds) {
        public Answer {
            probabilities = Collections.unmodifiableMap(new LinkedHashMap<>(probabilities));
            usage = Collections.unmodifiableMap(new LinkedHashMap<>(usage));
        }
    }

    /** A parsed response body, the attempts it took and the time spent waiting between them. */
    protected record Posted(ObjectNode body, int attempts, double waitedSeconds) {
    }

    /**
     * @param baseUrl the server origin, for example {@code https://api.typesafe.ai}
     * @param model   the model name to request
     * @param apiKey  sent as {@code Authorization: Bearer <key>}; {@code null} sends none
     * @param options thresholds, timeout, retries, transport and sleeper
     * @param defaultTimeoutSeconds the timeout when {@code options} sets none
     * @param defaultMaxRetries     the retries when {@code options} sets none
     * @param proxy   whether a default transport may use the system proxy settings
     */
    protected SystemOneDecisionModel(String baseUrl, String model, String apiKey, Builder<?> options,
                                     double defaultTimeoutSeconds, int defaultMaxRetries, boolean proxy) {
        this.functionThreshold = threshold(options.functionThreshold != null ? options.functionThreshold
                : options.threshold);
        this.argumentsThreshold = threshold(options.argumentsThreshold != null ? options.argumentsThreshold
                : options.threshold);
        double timeout = options.timeoutSeconds != null ? options.timeoutSeconds : defaultTimeoutSeconds;
        int retries = options.maxRetries != null ? options.maxRetries : defaultMaxRetries;
        if (retries < 0) {
            throw new IllegalArgumentException("maxRetries must be a nonnegative integer.");
        }
        if (Double.isNaN(timeout) || Double.isInfinite(timeout) || timeout <= 0) {
            throw new IllegalArgumentException("timeout must be a positive number of seconds.");
        }
        if (apiKey != null && (Py.strip(apiKey).isEmpty() || apiKey.contains("\n") || apiKey.contains("\r"))) {
            throw new IllegalArgumentException("API keys must be nonempty single-line strings.");
        }
        if (model == null || Py.strip(model).isEmpty()) {
            throw new IllegalArgumentException(getClass().getSimpleName() + " needs a model name.");
        }
        this.baseUrl = Py.baseUrl(baseUrl);
        if (this.baseUrl == null) {
            throw new IllegalArgumentException(getClass().getSimpleName()
                    + " needs an http(s) baseUrl without credentials, query or fragment.");
        }
        this.model = model;
        this.timeoutSeconds = timeout;
        this.maxRetries = retries;
        Map<String, String> headers = new LinkedHashMap<>();
        if (apiKey != null) {
            headers.put("Authorization", "Bearer " + apiKey);
        }
        headers.put("Content-Type", "application/json");
        this.headers = Collections.unmodifiableMap(headers);
        this.transport = options.transport != null ? options.transport
                : new JdkHttpTransport(Duration.ofMillis(Math.round(timeout * 1000)), proxy);
        this.ownsTransport = options.transport == null;
        this.sleeper = options.sleeper != null ? options.sleeper : Sleeper.SYSTEM;
    }

    /** A decision model at {@code baseUrl} that serves {@code model}. */
    public static GenericBuilder builder(String baseUrl, String model) {
        return new GenericBuilder(baseUrl, model);
    }

    private static double threshold(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value) || value < 0 || value > 1) {
            throw new IllegalArgumentException("Thresholds must be numbers between 0 and 1.");
        }
        return value;
    }

    /** The name in error messages. */
    protected String displayName() {
        return "Decision model";
    }

    /** The {@code source} field of the feedback records the generator reads. */
    protected String feedbackSource() {
        return "decision_model_noul_and_deterministic_schema_validation";
    }

    public double functionThreshold() {
        return functionThreshold;
    }

    public double argumentsThreshold() {
        return argumentsThreshold;
    }

    public String baseUrl() {
        return baseUrl;
    }

    /** The model name sent with each request. */
    public String model() {
        return model;
    }

    public int maxRetries() {
        return maxRetries;
    }

    protected final HttpTransport transport() {
        return transport;
    }

    protected final Map<String, String> requestHeaders() {
        return headers;
    }

    protected final Duration timeout() {
        return Duration.ofMillis(Math.round(timeoutSeconds * 1000));
    }

    @Override
    public void close() {
        if (ownsTransport) {
            transport.close();
        }
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(model=" + Py.strRepr(model) + ", baseUrl=" + Py.strRepr(baseUrl)
                + ", functionThreshold=" + functionThreshold + ", argumentsThreshold=" + argumentsThreshold + ")";
    }

    // HTTP --------------------------------------------------------------------------------
    /** A narrow diagnostic ID, never an echoed credential. */
    protected String requestId(HttpTransport.Response response) {
        String authorization = null;
        for (Map.Entry<String, String> header : headers.entrySet()) {
            if (header.getKey().equalsIgnoreCase("authorization")) {
                String value = header.getValue();
                authorization = value.startsWith("Bearer ") ? value.substring("Bearer ".length()) : value;
            }
        }
        for (String name : new String[]{"x-request-id", "request-id", "apim-request-id"}) {
            String value = response.header(name);
            if (value != null && !value.isEmpty() && SAFE_ID.matcher(value).matches()
                    && (authorization == null || !value.contains(authorization))) {
                return value;
            }
        }
        return null;
    }

    /** The error for a non-success response. */
    protected DecisionModelException error(HttpTransport.Response response, int attempts) {
        return new DecisionModelException(displayName() + " HTTP " + response.status() + "; no judgment recorded.",
                "http_error", response.status(), requestId(response), attempts);
    }

    private double pause(double seconds) {
        long started = System.nanoTime();
        try {
            sleeper.sleep(seconds);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DecisionModelException(displayName() + " was interrupted while waiting to retry.");
        }
        return (System.nanoTime() - started) / 1e9;
    }

    static double retryDelay(HttpTransport.Response response, int attempt) {
        if (response != null) {
            String value = response.header("Retry-After");
            if (value != null) {
                Double parsed = Py.parseFloat(value);
                double delay;
                if (parsed != null) {
                    delay = parsed;
                } else {
                    try {
                        ZonedDateTime stamp = ZonedDateTime.parse(Py.strip(value), DateTimeFormatter.RFC_1123_DATE_TIME);
                        delay = (stamp.toInstant().toEpochMilli() - System.currentTimeMillis()) / 1000.0;
                    } catch (DateTimeParseException e) {
                        delay = -1;
                    }
                }
                if (!Double.isNaN(delay) && !Double.isInfinite(delay) && delay >= 0) {
                    return Math.min(delay, 30.0);
                }
            }
        }
        return Math.min(Math.pow(2, attempt), 30.0);
    }

    /** POST with bounded retries. */
    protected Posted post(ObjectNode payload) {
        byte[] body = Json.compact(payload).getBytes(StandardCharsets.UTF_8);
        double waited = 0.0;
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            HttpTransport.Response response;
            try {
                response = transport.send(new HttpTransport.Request("POST", baseUrl + "/v1/systemone", headers, body,
                        timeout()));
            } catch (IOException e) {
                if (attempt < maxRetries) {
                    waited += pause(retryDelay(null, attempt));
                    continue;
                }
                throw new DecisionModelException(displayName() + " connection failed after " + (attempt + 1)
                        + " attempts.", "connection_failed", null, null, attempt + 1);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new DecisionModelException(displayName() + " request was interrupted.", "connection_failed");
            }
            if (RETRY_STATUSES.contains(response.status()) && attempt < maxRetries) {
                waited += pause(retryDelay(response, attempt));
                continue;
            }
            if (!response.isSuccess()) {
                throw error(response, attempt + 1);
            }
            JsonNode parsed;
            try {
                parsed = Json.parse(response.body());
            } catch (IllegalArgumentException e) {
                throw new DecisionModelException(displayName() + " response was not valid JSON.", "response_invalid");
            }
            if (!parsed.isObject()) {
                throw new DecisionModelException(displayName() + " response must be a JSON object.",
                        "response_invalid");
            }
            return new Posted(checked((ObjectNode) parsed), attempt + 1, waited);
        }
        throw new IllegalStateException("unreachable");
    }

    /** Hook for a subclass to check the response's model identity. */
    protected ObjectNode checked(ObjectNode body) {
        return body;
    }

    // The protocol ------------------------------------------------------------------------
    /**
     * Asks {@code questions} about {@code state}; returns the probabilities of yes, the model, usage
     * and timing. {@code state} holds {@code messages}, {@code tools} and {@code proposed_calls}.
     * Override this method to connect a decision model that speaks another protocol.
     */
    public Answer ask(ObjectNode state, ObjectNode questions) {
        ObjectNode payload = Json.object();
        payload.put("model", model);
        payload.set("state", state);
        payload.set("questions", questions);
        Posted posted = post(payload);
        ObjectNode body = posted.body();
        JsonNode answers = body.get("answers");
        Map<String, Double> probabilities = new LinkedHashMap<>();
        for (Iterator<String> it = questions.fieldNames(); it.hasNext(); ) {
            String key = it.next();
            JsonNode answer = answers != null && answers.isObject() ? answers.get(key) : null;
            if (answer == null || !answer.isObject() || !"noul".equals(Py.text(answer.get("type")))) {
                throw new DecisionModelException(displayName() + " response is missing the answer " + Py.strRepr(key)
                        + ".", "response_invalid");
            }
            JsonNode value = answer.get("noul");
            if (value == null || !value.isNumber() || !Json.isFinite(value) || value.doubleValue() < 0
                    || value.doubleValue() > 1) {
                throw new DecisionModelException(displayName() + " returned an invalid probability for "
                        + Py.strRepr(key) + ".", "response_invalid");
            }
            probabilities.put(key, value.doubleValue());
        }
        String resolved = Py.text(body.get("model"));
        if (resolved == null || resolved.isEmpty()) {
            throw new DecisionModelException(displayName() + " response is missing the resolved model ID.",
                    "response_invalid");
        }
        JsonNode usage = body.get("usage");
        if (usage == null || !usage.isObject()) {
            throw new DecisionModelException(displayName() + " response is missing token usage.", "response_invalid");
        }
        Map<String, Long> counts = new LinkedHashMap<>();
        for (String kind : new String[]{"input_tokens", "output_tokens"}) {
            JsonNode count = usage.get(kind);
            if (Py.isNone(count)) {
                counts.put(kind, null);
            } else if (!count.isIntegralNumber() || count.bigIntegerValue().signum() < 0 || !count.canConvertToLong()) {
                throw new DecisionModelException(displayName() + " response contains invalid token usage.",
                        "response_invalid");
            } else {
                counts.put(kind, count.longValue());
            }
        }
        return new Answer(probabilities, resolved, counts, posted.attempts(), posted.waitedSeconds());
    }

    @Override
    public Judgment judge(ArrayNode messages, ArrayNode tools, List<Call> calls) {
        long started = System.nanoTime();
        SchemaReport local = SchemaCheck.checkJson(tools, Calls.toJson(calls));
        boolean argumentsSkipped = calls != null && calls.isEmpty();
        Double functionProbability = null;
        Double argumentsProbability = null;
        Map<String, Long> usage = new LinkedHashMap<>();
        usage.put("input_tokens", 0L);
        usage.put("output_tokens", 0L);
        String resolvedModel = null;
        int apiCalls = 0;
        double waited = 0.0;
        if (local.wellFormed()) {  // an unreadable candidate is not sent to the decision model
            if (messages == null) {
                throw new IllegalArgumentException("messages must be a list");
            }
            ArrayNode observable = Json.array();
            for (JsonNode message : messages) {
                if (!message.isObject()) {
                    throw new IllegalArgumentException("Each message must be an object");
                }
                ObjectNode copy = observable.addObject();
                for (String key : OBSERVABLE_KEYS) {
                    if (message.has(key)) {
                        copy.set(key, message.get(key).deepCopy());
                    }
                }
            }
            ObjectNode state = Json.object();
            state.set("messages", observable);
            state.set("tools", SchemaCheck.functionDefinitionsJson(tools));
            ArrayNode proposed = state.putArray("proposed_calls");
            for (Call call : calls) {
                ObjectNode item = proposed.addObject();
                item.put("name", call.name());
                item.set("arguments", call.arguments().deepCopy());
            }
            ObjectNode questions = Json.object();
            questions.set("function_correct", Prompts.functionQuestion());
            if (!argumentsSkipped) {
                questions.set("arguments_correct", Prompts.argumentQuestion());
            }
            Answer answer = ask(state, questions);
            functionProbability = answer.probabilities().get("function_correct");
            if (!argumentsSkipped) {
                argumentsProbability = answer.probabilities().get("arguments_correct");
            }
            resolvedModel = answer.model();
            usage = new LinkedHashMap<>(answer.usage());
            apiCalls = answer.apiCalls();
            waited = answer.retryWaitSeconds();
        }

        // V(p) is true iff p is not null and p >= threshold; S must pass too.
        boolean functionPass = local.functionPass() && functionProbability != null
                && functionProbability >= functionThreshold;
        Boolean argumentsPass = argumentsSkipped ? null : (local.argumentsPass() && argumentsProbability != null
                && argumentsProbability >= argumentsThreshold);
        ObjectNode checks = Json.object();
        ObjectNode function = checks.putObject("function_selection");
        function.put("pass", functionPass);
        function.put("probability_yes", functionProbability);
        function.put("threshold", functionThreshold);
        function.put("schema_pass", local.functionPass());
        ObjectNode arguments = checks.putObject("arguments");
        arguments.put("skipped", argumentsSkipped);
        arguments.put("pass", argumentsPass);
        arguments.put("probability_yes", argumentsProbability);
        arguments.put("threshold", argumentsThreshold);
        arguments.put("schema_pass", local.argumentsPass());
        ObjectNode feedback = Json.object();
        feedback.put("source", feedbackSource());
        feedback.set("checks", checks);
        ArrayNode failed = feedback.putArray("failed_checks");
        if (!functionPass) {
            failed.add("function_selection");
        }
        if (!argumentsSkipped && !Boolean.TRUE.equals(argumentsPass)) {
            failed.add("arguments");
        }
        feedback.set("schema_errors", local.errorsJson());
        feedback.put("instruction", Prompts.REPAIR_INSTRUCTION);
        return Judgment.builder()
                .functionPass(functionPass)
                .argumentsPass(argumentsPass)
                .argumentsSkipped(argumentsSkipped)
                .functionProbability(functionProbability)
                .argumentsProbability(argumentsProbability)
                .accepted(functionPass && (argumentsSkipped || Boolean.TRUE.equals(argumentsPass)))
                .feedback(feedback)
                .schemaErrors(local.errors())
                .usage(usage)
                .model(resolvedModel)
                .requestedModel(model)
                .latencySeconds(Math.max(0.0, (System.nanoTime() - started) / 1e9 - waited))
                .retryWaitSeconds(waited)
                .apiCalls(apiCalls)
                .build();
    }

    /**
     * Options shared by every decision model over this protocol.
     *
     * @param <B> the builder type
     */
    public abstract static class Builder<B extends Builder<B>> {
        protected double threshold = 0.5;
        protected Double functionThreshold;
        protected Double argumentsThreshold;
        protected Double timeoutSeconds;
        protected Integer maxRetries;
        protected HttpTransport transport;
        protected Sleeper sleeper;

        protected abstract B self();

        /** The pass mark of both questions (0.5 by default, the paper's setting). */
        public B threshold(double threshold) {
            this.threshold = threshold;
            return self();
        }

        /** Overrides the function question's pass mark. */
        public B functionThreshold(Double functionThreshold) {
            this.functionThreshold = functionThreshold;
            return self();
        }

        /** Overrides the argument question's pass mark. */
        public B argumentsThreshold(Double argumentsThreshold) {
            this.argumentsThreshold = argumentsThreshold;
            return self();
        }

        public B timeout(Duration timeout) {
            this.timeoutSeconds = timeout.toMillis() / 1000.0;
            return self();
        }

        /** HTTP retries after the first attempt. */
        public B maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
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

    /** Builds a {@link SystemOneDecisionModel} for any server of the protocol. */
    public static final class GenericBuilder extends Builder<GenericBuilder> {
        private final String baseUrl;
        private final String model;
        private String apiKey;

        GenericBuilder(String baseUrl, String model) {
            this.baseUrl = baseUrl;
            this.model = model;
        }

        public GenericBuilder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        @Override
        protected GenericBuilder self() {
            return this;
        }

        public SystemOneDecisionModel build() {
            return new SystemOneDecisionModel(baseUrl, model, apiKey, this, 60.0, 3, true);
        }
    }
}
