package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.Iterator;
import java.util.Map;

/**
 * One proposed function call.
 *
 * @param name      the function name ({@code null} if the candidate named none)
 * @param arguments the arguments, normally a JSON object; {@code null} when the model produced
 *                  arguments that are not valid JSON (a malformed call, rejected and repaired)
 */
public record Call(String name, JsonNode arguments) implements JsonConvertible {
    public Call {
        arguments = arguments == null || arguments.isNull() || arguments.isMissingNode() ? null : arguments.deepCopy();
    }

    /** A call from a name and arguments given as a map, JSON or any value {@link Json#node} accepts. */
    public static Call of(String name, Object arguments) {
        return new Call(name, arguments == null ? null : Json.node(arguments));
    }

    /** The arguments as an object, or {@code null} if they are missing or not an object. */
    public ObjectNode argumentsObject() {
        return arguments != null && arguments.isObject() ? (ObjectNode) arguments.deepCopy() : null;
    }

    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        if (name == null) {
            node.putNull("name");
        } else {
            node.put("name", name);
        }
        node.set("arguments", arguments == null ? NullNode.getInstance() : arguments.deepCopy());
        return node;
    }

    /** {@code get_weather(city="Seoul")}. */
    @Override
    public String toString() {
        if (arguments == null || !arguments.isObject()) {
            return name + "(<malformed arguments>)";
        }
        StringBuilder sb = new StringBuilder(String.valueOf(name)).append('(');
        boolean first = true;
        for (Iterator<Map.Entry<String, JsonNode>> it = arguments.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            if (!first) {
                sb.append(", ");
            }
            first = false;
            sb.append(entry.getKey()).append('=').append(Json.dumps(entry.getValue()));
        }
        return sb.append(')').toString();
    }
}
