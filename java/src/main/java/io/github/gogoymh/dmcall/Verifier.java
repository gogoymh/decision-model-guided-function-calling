package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.node.ArrayNode;

import java.util.List;

/**
 * Checks calls from any source before running them; nothing is generated or executed.
 *
 * <p>The calls pass if they pass the schema check and the decision model's function question and,
 * unless there is no call, its argument question.
 */
public final class Verifier {
    private Verifier() {
    }

    /**
     * Verifies {@code calls} with {@code decisionModel}.
     *
     * @param messages      the conversation so far, including earlier tool calls and tool results
     * @param calls         {@link Call}s or calls in any SDK shape ({@code null} means no call)
     * @param tools         {@link Tool}s or tool definitions
     * @param decisionModel the decision model, configured with its own key and thresholds
     */
    public static Verdict verify(Object messages, Object calls, Object tools, DecisionModel decisionModel) {
        return verdict(decisionModel, messages, calls, tools);
    }

    /** Verifies with Jev ({@code TYPESAFE_API_KEY}) at the paper's threshold (0.5). */
    public static Verdict verify(Object messages, Object calls, Object tools) {
        return verify(messages, calls, tools, DMCall.DEFAULT_THRESHOLD, null);
    }

    /** Verifies with Jev at {@code threshold}, with {@code jevApiKey} or {@code TYPESAFE_API_KEY}. */
    public static Verdict verify(Object messages, Object calls, Object tools, double threshold, String jevApiKey) {
        try (Jev jev = Jev.builder().apiKey(jevApiKey).threshold(threshold).build()) {
            return verdict(jev, messages, calls, tools);
        }
    }

    static Verdict verdict(DecisionModel decisionModel, Object messages, Object calls, Object tools) {
        if (decisionModel == null) {
            throw new IllegalArgumentException("A decision model is required.");
        }
        Tools.Toolset toolset = Tools.toolset(tools, null);
        ArrayNode conversation = Messages.conversation(messages);
        List<Call> proposed = Calls.normalize(calls);
        SchemaReport local = SchemaCheck.checkJson(toolset.schemas(), Calls.toJson(proposed));
        Judgment check = decisionModel.judge(conversation, toolset.schemas(), proposed);
        boolean accepted = local.accepted() && Boolean.TRUE.equals(check.functionPass())
                && (check.argumentsSkipped() || Boolean.TRUE.equals(check.argumentsPass()));
        return new Verdict(accepted, proposed, check.functionProbability(), check.argumentsProbability(),
                local.errors(), check.feedback(), check);
    }
}
