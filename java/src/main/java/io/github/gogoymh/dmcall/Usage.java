package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Token counts summed over every generator request and every decision-model request. Unknown
 * counts are left out, never counted as zero.
 *
 * @param generator     for example {@code input_tokens}, {@code cached_input_tokens}, {@code output_tokens}
 * @param decisionModel {@code input_tokens} and {@code output_tokens}
 */
public record Usage(Map<String, Long> generator, Map<String, Long> decisionModel) implements JsonConvertible {
    public Usage {
        generator = Collections.unmodifiableMap(new LinkedHashMap<>(generator));
        decisionModel = Collections.unmodifiableMap(new LinkedHashMap<>(decisionModel));
    }

    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.set("generator", Proposal.usageJson(generator));
        node.set("decision_model", Proposal.usageJson(decisionModel));
        return node;
    }
}
