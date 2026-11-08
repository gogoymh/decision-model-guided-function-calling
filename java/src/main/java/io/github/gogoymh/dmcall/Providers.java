package io.github.gogoymh.dmcall;

import java.util.List;
import java.util.Locale;

/** Providers from {@code "provider:model"} specs, such as {@code "openai:gpt-5-nano"}. */
public final class Providers {
    /** The provider names a spec may use. */
    public static final List<String> NAMES = List.of("anthropic", "azure", "claude", "gemini", "google", "grok",
            "meta", "openai", "openrouter", "xai");

    private Providers() {
    }

    /**
     * A provider from {@code "provider:model"}, configured from environment variables: for example
     * {@code "openai:gpt-5-nano"}, {@code "anthropic:claude-sonnet-4-5"}, {@code "gemini:gemini-2.5-flash"},
     * {@code "xai:grok-4"}, {@code "meta:muse-spark-1.3"}, {@code "openrouter:meta/llama"} or
     * {@code "azure:<deployment>"}.
     */
    public static Provider fromSpec(String spec) {
        return builder(spec).build();
    }

    /** The builder for a spec, to set a key, endpoint, transport or other options before building. */
    public static Provider.Builder<?, ?> builder(String spec) {
        int colon = spec == null ? -1 : spec.indexOf(':');
        if (colon < 0 || colon == spec.length() - 1) {
            throw new IllegalArgumentException("Use \"provider:model\", e.g. \"openai:gpt-5-nano\", "
                    + "\"anthropic:claude-sonnet-4-5\", \"gemini:gemini-2.5-flash\", \"xai:grok-4\", "
                    + "\"meta:muse-spark-1.3\" or \"azure:<deployment>\".");
        }
        String name = spec.substring(0, colon).toLowerCase(Locale.ROOT);
        String model = spec.substring(colon + 1);
        switch (name) {
            case "openai":
                return OpenAI.builder(model);
            case "azure":
                return AzureOpenAI.builder(model);
            case "anthropic":
            case "claude":
                return Anthropic.builder(model);
            case "gemini":
            case "google":
                return Gemini.builder(model);
            case "xai":
            case "grok":
                return XAI.builder(model);
            case "openrouter":
                return OpenRouter.builder(model);
            case "meta":
                return Meta.builder(model);
            default:
                throw new IllegalArgumentException("Unknown provider " + Py.strRepr(spec.substring(0, colon))
                        + "; choose from " + String.join(", ", NAMES)
                        + ", or use OpenAICompatible.builder(model).baseUrl(...) for any other endpoint.");
        }
    }
}
