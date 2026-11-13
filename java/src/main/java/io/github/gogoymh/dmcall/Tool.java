package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A tool the model may call: a JSON Schema function definition, optionally paired with the
 * {@link ToolHandler} that runs it.
 *
 * <p>Definitions may be in OpenAI (Chat Completions or Responses), Anthropic
 * ({@code input_schema}) or Gemini ({@code parameters} or {@code parametersJsonSchema}, upper-case
 * types allowed) format; they are stored as {@code {"type": "function", "function": {...}}} and
 * checked when the tool is created.
 */
public final class Tool implements JsonConvertible {
    private final ObjectNode definition;
    private final ToolHandler handler;

    private Tool(ObjectNode definition, ToolHandler handler) {
        this.definition = definition;
        this.handler = handler;
    }

    /** A tool from a definition given as a {@link JsonNode}, a {@code Map} or another {@link Tool}. */
    public static Tool of(Object definition) {
        return of(definition, null);
    }

    /** A tool from a definition, run by {@code handler}. */
    public static Tool of(Object definition, ToolHandler handler) {
        if (definition instanceof Tool) {
            return new Tool(((Tool) definition).definition.deepCopy(), handler);
        }
        JsonNode node = Json.node(definition);
        if (!node.isObject()) {
            throw new IllegalArgumentException("A tool is a function definition object.");
        }
        return new Tool(Tools.schemaOf(node), handler);
    }

    /** A tool from a name, a description and a JSON Schema for its parameters ({@code null} for none). */
    public static Tool function(String name, String description, Object parameters) {
        return function(name, description, parameters, null);
    }

    /** {@link #function(String, String, Object)} run by {@code handler}. */
    public static Tool function(String name, String description, Object parameters, ToolHandler handler) {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", name);
        function.put("description", description == null ? "" : description);
        if (parameters != null) {
            function.put("parameters", parameters);
        }
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", function);
        return of(tool, handler);
    }

    /** The same definition run by {@code handler}. */
    public Tool withHandler(ToolHandler handler) {
        return new Tool(definition.deepCopy(), handler);
    }

    public String name() {
        return definition.get("function").get("name").textValue();
    }

    public String description() {
        return definition.get("function").get("description").textValue();
    }

    /** A copy of the parameter schema. */
    public ObjectNode parameters() {
        return (ObjectNode) definition.get("function").get("parameters").deepCopy();
    }

    /** A copy of the definition, {@code {"type": "function", "function": {...}}}. */
    public ObjectNode definition() {
        return definition.deepCopy();
    }

    /** The handler, or {@code null}. */
    public ToolHandler handler() {
        return handler;
    }

    @Override
    public JsonNode toJson() {
        return definition.deepCopy();
    }

    @Override
    public String toString() {
        return "Tool(" + name() + (handler == null ? "" : ", with handler") + ")";
    }
}
