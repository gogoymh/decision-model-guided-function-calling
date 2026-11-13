package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Feedback representations: how the decision model's scores reach the generator.
 *
 * <p>Only the generator's copy of earlier feedback changes. Acceptance always uses the numeric
 * probabilities, and {@link Result#trace()} keeps them unchanged.
 * <ul>
 *   <li>{@link FeedbackMode#HYBRID} (the default): each probability keeps its value and threshold
 *       and gains a verbal label.</li>
 *   <li>{@link FeedbackMode#NUMERIC}: probabilities and thresholds only.</li>
 *   <li>{@link FeedbackMode#TEXT}: labels only; probabilities and thresholds are removed.</li>
 * </ul>
 * The label uses five bands of equal width, lower bound inclusive; the last band includes 1. A
 * question that was not asked (the argument question on a no call) keeps a null score and label.
 */
public final class Feedback {
    private Feedback() {
    }

    /** The five verbal bands, lowest first. */
    public static final List<String> LABELS = List.of("very likely incorrect", "likely incorrect", "uncertain",
            "likely correct", "very likely correct");
    /** The lower bounds of bands two to five. */
    public static final List<Double> CUT_POINTS = List.of(0.2, 0.4, 0.6, 0.8);

    /** Where a probability's label goes. Decision-model records use {@code probability_yes}. */
    static final Map<String, String> LABEL_FIELDS = labelFields();
    static final Set<String> THRESHOLDS = Set.of("threshold", "function_threshold", "arguments_threshold");

    private static Map<String, String> labelFields() {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("probability_yes", "feedback_label");
        fields.put("function_probability", "function_label");
        fields.put("arguments_probability", "arguments_label");
        fields.put("p_f", "function_label");
        fields.put("p_a", "arguments_label");
        return java.util.Collections.unmodifiableMap(fields);
    }

    /** The verbal band of a probability; {@code null} for a question that was not asked. */
    public static String label(Double probability) {
        if (probability == null) {
            return null;
        }
        double p = probability;
        if (Double.isNaN(p) || Double.isInfinite(p) || p < 0 || p > 1) {
            throw new IllegalArgumentException(
                    "Feedback probabilities must be finite numbers between 0 and 1, or null.");
        }
        int band = 0;
        for (double cut : CUT_POINTS) {
            if (cut <= p) {
                band++;
            }
        }
        return LABELS.get(band);
    }

    /** The label of a JSON probability: a number, or null. Booleans and text are rejected. */
    public static String label(JsonNode probability) {
        if (Py.isNone(probability)) {
            return null;
        }
        if (!probability.isNumber()) {
            throw new IllegalArgumentException(
                    "Feedback probabilities must be finite numbers between 0 and 1, or null.");
        }
        return label(probability.doubleValue());
    }

    private static JsonNode labelNode(JsonNode probability) {
        String label = label(probability);
        return label == null ? NullNode.getInstance() : TextNode.valueOf(label);
    }

    /** The generator's copy of one feedback record under a representation. The input is not changed. */
    public static ObjectNode represent(JsonNode feedback, FeedbackMode representation) {
        if (representation == null) {
            throw new IllegalArgumentException("feedback must be one of 'hybrid', 'numeric', 'text'.");
        }
        if (feedback == null || !feedback.isObject()) {
            throw new IllegalArgumentException("Repair feedback must be an object.");
        }
        switch (representation) {
            case NUMERIC:
                return (ObjectNode) feedback.deepCopy();
            case TEXT:
                return (ObjectNode) text(feedback);
            default:
                return (ObjectNode) hybrid(feedback);
        }
    }

    /** {@link #represent(JsonNode, FeedbackMode)} with the representation's name. */
    public static ObjectNode represent(JsonNode feedback, String representation) {
        return represent(feedback, FeedbackMode.fromValue(representation));
    }

    /** Every record of a repair history under a representation, in the same order. */
    public static ArrayNode representHistory(JsonNode history, FeedbackMode representation) {
        if (representation == null) {
            throw new IllegalArgumentException("feedback must be one of 'hybrid', 'numeric', 'text'.");
        }
        ArrayNode result = Json.array();
        if (history == null) {
            return result;
        }
        for (JsonNode item : history) {
            if (item == null || !item.isObject() || item.get("feedback") == null || !item.get("feedback").isObject()) {
                throw new IllegalArgumentException("Each repair history record needs a feedback object.");
            }
            ObjectNode copy = (ObjectNode) item.deepCopy();
            copy.set("feedback", represent(item.get("feedback"), representation));
            result.add(copy);
        }
        return result;
    }

    /**
     * A feedback record without the other question, for DM-Call-Function ({@code keep="function"})
     * and DM-Call-Argument ({@code keep="arguments"}). The source, schema errors, thresholds and
     * instruction stay; only the hidden question's verdict and score are removed.
     */
    public static ObjectNode singleCheck(JsonNode feedback, String keep) {
        if (!"function".equals(keep) && !"arguments".equals(keep)) {
            throw new IllegalArgumentException("keep must be 'function' or 'arguments'.");
        }
        if (feedback == null || !feedback.isObject()) {
            throw new IllegalArgumentException("Repair feedback must be an object.");
        }
        String hidden = keep.equals("function") ? "arguments" : "function";
        String hiddenRecord = hidden.equals("function") ? "function_selection" : "arguments";
        ObjectNode result = (ObjectNode) feedback.deepCopy();
        JsonNode checks = result.get("checks");
        if (checks != null && checks.isObject()) {
            ((ObjectNode) checks).remove(hiddenRecord);
        }
        JsonNode failed = result.get("failed_checks");
        if (failed != null && failed.isArray()) {
            ArrayNode kept = Json.array();
            for (JsonNode name : failed) {
                if (!hiddenRecord.equals(Py.text(name))) {
                    kept.add(name);
                }
            }
            result.set("failed_checks", kept);
        }
        List<String> drop = new ArrayList<>();
        for (Iterator<String> it = result.fieldNames(); it.hasNext(); ) {
            String key = it.next();
            if (key.startsWith(hidden + "_")) {
                drop.add(key);
            }
        }
        result.remove(drop);
        return result;
    }

    /** {@link #singleCheck(JsonNode, String)} for a {@link Checks} variant other than {@code BOTH}. */
    public static ObjectNode singleCheck(JsonNode feedback, Checks keep) {
        return singleCheck(feedback, keep == null ? null : (keep == Checks.BOTH ? "both" : keep.value()));
    }

    private static JsonNode hybrid(JsonNode value) {
        if (value.isArray()) {
            ArrayNode result = Json.array();
            for (JsonNode item : value) {
                result.add(hybrid(item));
            }
            return result;
        }
        if (!value.isObject()) {
            return value.deepCopy();
        }
        ObjectNode result = Json.object();
        for (Iterator<Map.Entry<String, JsonNode>> it = value.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            JsonNode item = entry.getValue();
            // Schema errors are diagnostics about the task input, not scores.
            result.set(key, key.equals("schema_errors") ? item.deepCopy() : hybrid(item));
            String field = LABEL_FIELDS.get(key);
            if (field != null) {
                if (value.has(field) || result.has(field)) {
                    throw new IllegalArgumentException("Repair feedback has conflicting added judgment fields.");
                }
                result.set(field, labelNode(item));
            }
        }
        return result;
    }

    private static JsonNode text(JsonNode value) {
        if (value.isArray()) {
            ArrayNode result = Json.array();
            for (JsonNode item : value) {
                result.add(text(item));
            }
            return result;
        }
        if (!value.isObject()) {
            return value.deepCopy();
        }
        ObjectNode result = Json.object();
        for (Iterator<Map.Entry<String, JsonNode>> it = value.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            JsonNode item = entry.getValue();
            if (THRESHOLDS.contains(key)) {
                continue;
            }
            String field = LABEL_FIELDS.get(key);
            if (field != null) {
                if (value.has(field) || result.has(field)) {
                    throw new IllegalArgumentException("Repair feedback has conflicting judgment fields.");
                }
                result.set(field, labelNode(item));
            } else if (key.equals("instruction") && item.isTextual()) {
                result.put(key, item.textValue().replace("The probabilities", "The feedback labels"));
            } else if (key.equals("schema_errors")) {
                result.set(key, item.deepCopy());
            } else {
                result.set(key, text(item));
            }
        }
        return result;
    }
}
