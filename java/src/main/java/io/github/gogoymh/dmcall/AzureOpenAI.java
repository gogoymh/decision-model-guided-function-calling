package io.github.gogoymh.dmcall;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * An Azure OpenAI deployment, directly or behind an API Management gateway.
 *
 * <p>Key: {@code AZURE_OPENAI_API_KEY} (or {@code AZURE_API_KEY}), endpoint:
 * {@code AZURE_OPENAI_ENDPOINT} (or {@code AZURE_ENDPOINT}), version:
 * {@code AZURE_OPENAI_API_VERSION} ({@code 2024-12-01-preview} by default).
 */
public final class AzureOpenAI extends ChatCompletionsProvider {
    static final Kind KIND = new Kind("azure", null, List.of(), List.of("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"), true);
    private static final Pattern VERSION = Pattern.compile("\\d{4}-\\d{2}-\\d{2}(?:-preview)?");

    private final String apiVersion;

    private AzureOpenAI(Builder b) {
        super(b, KIND, "max_completion_tokens", endpoint(b));
        this.apiVersion = version(b);
    }

    /** A provider for {@code deployment} with the endpoint and key from the environment. */
    public AzureOpenAI(String deployment) {
        this(builder(deployment));
    }

    public static Builder builder(String deployment) {
        return new Builder(deployment);
    }

    private static String endpoint(Builder b) {
        String endpoint = b.endpoint != null && !b.endpoint.isEmpty() ? b.endpoint
                : Env.first(List.of("AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"));
        if (endpoint == null) {
            throw new IllegalArgumentException("AzureOpenAI needs endpoint(...) or AZURE_OPENAI_ENDPOINT.");
        }
        String version = version(b);
        if (!VERSION.matcher(version).matches()) {
            throw new IllegalArgumentException("apiVersion must look like 2024-12-01-preview.");
        }
        String deployment = b.model;
        boolean invalid = deployment == null || deployment.equals(".") || deployment.equals("..")
                || deployment.chars().anyMatch(c -> c < 32);
        if (invalid) {
            throw new IllegalArgumentException("The deployment name contains invalid characters.");
        }
        return endpoint;
    }

    private static String version(Builder b) {
        if (b.apiVersion != null && !b.apiVersion.isEmpty()) {
            return b.apiVersion;
        }
        String fromEnv = Env.first(List.of("AZURE_OPENAI_API_VERSION"));
        return fromEnv != null ? fromEnv : "2024-12-01-preview";
    }

    public String apiVersion() {
        return apiVersion;
    }

    @Override
    public String url() {
        return baseUrl + "/openai/deployments/" + Py.quote(model) + "/chat/completions?api-version=" + apiVersion;
    }

    @Override
    protected Map<String, String> authHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("api-key", apiKey());
        return headers;
    }

    /** Builds an {@link AzureOpenAI}. */
    public static final class Builder extends ChatBuilder<Builder, AzureOpenAI> {
        private String endpoint;
        private String apiVersion;

        Builder(String deployment) {
            super(deployment);
        }

        /** The resource endpoint, for example {@code https://my-resource.openai.azure.com}. */
        public Builder endpoint(String endpoint) {
            this.endpoint = endpoint;
            return this;
        }

        /** The API version ({@code 2024-12-01-preview} by default). */
        public Builder apiVersion(String apiVersion) {
            this.apiVersion = apiVersion;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public AzureOpenAI build() {
            return new AzureOpenAI(this);
        }
    }
}
