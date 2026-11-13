package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;

/** A value with a JSON form, used wherever DM-Call accepts JSON-like input. */
public interface JsonConvertible {
    /** A new JSON tree for this value. */
    JsonNode toJson();
}
