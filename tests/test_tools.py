"""Tools from Python functions, tool definitions in every format, and calls from any SDK."""

import enum
from typing import Annotated, Literal, Optional, TypedDict

import pytest

import dmcall
from conftest import WEATHER, ScriptedJev, ScriptedModel


class Unit(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class Address(TypedDict):
    street: str
    zip_code: Annotated[str, "Postal code"]


def book(city: str, guests: int, budget: float, refundable: bool, tags: list[str], extras: dict[str, int],
         room: Literal["single", "double"], unit: Unit = Unit.CELSIUS, note: Optional[str] = None,
         when: tuple[int, int] = (9, 0), address: Address | None = None,
         label: Annotated[str, "A short label"] = "trip", **more) -> dict:
    """Book a stay.

    Longer description.

    Args:
        city: Where to stay.
        guests (int): How many people,
            including children.
        room: Room type.

    Returns:
        A confirmation.
    """
    return {"city": city, "unit": unit}


def test_schema_from_python_function():
    schema = dmcall.schema_of(book)["function"]
    properties = schema["parameters"]["properties"]
    assert schema["name"] == "book" and schema["description"] == "Book a stay.\n\nLonger description."
    assert schema["parameters"]["required"] == ["city", "guests", "budget", "refundable", "tags", "extras", "room"]
    assert schema["parameters"]["additionalProperties"] is True
    assert properties["city"] == {"type": "string", "description": "Where to stay."}
    assert properties["guests"] == {"type": "integer", "description": "How many people, including children."}
    assert properties["budget"] == {"type": "number"} and properties["refundable"] == {"type": "boolean"}
    assert properties["tags"] == {"type": "array", "items": {"type": "string"}}
    assert properties["extras"] == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert properties["room"] == {"type": "string", "enum": ["single", "double"], "description": "Room type."}
    assert properties["unit"] == {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    assert properties["note"] == {"type": "string", "default": None}
    assert properties["when"] == {"type": "array", "prefixItems": [{"type": "integer"}] * 2, "minItems": 2,
                                  "maxItems": 2, "default": [9, 0]}
    assert properties["address"]["properties"]["zip_code"] == {"type": "string", "description": "Postal code"}
    assert properties["address"]["required"] == ["street", "zip_code"] and properties["address"]["default"] is None
    assert properties["label"] == {"type": "string", "description": "A short label", "default": "trip"}
    arguments = {"city": "Seoul", "guests": 2, "budget": 1.5, "refundable": True, "tags": [], "extras": {},
                 "room": "single", "unit": "fahrenheit"}
    assert dmcall.invoke(book, arguments)["unit"] is Unit.FAHRENHEIT


def test_docstring_styles():
    def numpy_style(city, days=1):
        """Forecast.

        Parameters
        ----------
        city : str
            City name.
        days : int
            Number of days.
        """

    def sphinx_style(city, days=1):
        """Forecast.

        :param city: City name.
        :param int days: Number of
            days.
        :returns: A forecast.
        """

    for function in (numpy_style, sphinx_style):
        properties = dmcall.schema_of(function)["function"]["parameters"]["properties"]
        assert dmcall.schema_of(function)["function"]["description"] == "Forecast."
        assert properties["city"]["description"] == "City name." and properties["days"]["description"] == "Number of days."
        assert properties["days"]["default"] == 1 and "type" not in properties["city"]


def test_tool_decorator_and_definition_formats():
    @dmcall.tool(name="weather", description="Weather now.")
    def get_weather(city: str):
        return city

    assert dmcall.schema_of(get_weather)["function"]["name"] == "weather"
    assert dmcall.schema_of(get_weather)["function"]["description"] == "Weather now."
    assert get_weather("Seoul") == "Seoul"
    custom = dmcall.tool(lambda **kw: kw, name="raw", parameters={"type": "object", "properties": {"q": {}}})
    assert dmcall.schema_of(custom)["function"]["parameters"] == {"type": "object", "properties": {"q": {}}}
    parameters = WEATHER["function"]["parameters"]
    for definition in (WEATHER, WEATHER["function"], {"type": "function", **WEATHER["function"]},
                       {"name": "get_weather", "description": "Current weather for a city.", "input_schema": parameters},
                       {"name": "get_weather", "description": "Current weather for a city.", "parameters": {
                           "type": "OBJECT", "properties": {"city": {"type": "STRING"},
                                                            "unit": {"type": "STRING", "enum": ["celsius", "fahrenheit"]}},
                           "required": ["city"]}}):
        assert dmcall.schema_of(definition) == WEATHER
    schemas, registry = dmcall.toolset([get_weather, WEATHER], functions={"get_weather": print})
    assert [s["function"]["name"] for s in schemas] == ["weather", "get_weather"]
    assert set(registry) == {"weather", "get_weather"}


def test_capital_types_are_lowered_only_in_schemas():
    gemini = {"name": "plan", "description": "", "parameters": {"type": "OBJECT", "properties": {
        "tier": {"type": "OBJECT", "enum": [{"type": "PREMIUM"}, {"type": "BASIC"}], "default": {"type": "PREMIUM"}},
        "kind": {"const": {"type": "FIXED"}, "examples": [{"type": "SAMPLE"}]},
        "default": {"type": "STRING"}, "enum": {"type": "ARRAY", "items": {"type": ["STRING", "NULL"]}},
        "type": {"type": "INTEGER"}}}}
    assert dmcall.schema_of(gemini)["function"]["parameters"] == {"type": "object", "properties": {
        "tier": {"type": "object", "enum": [{"type": "PREMIUM"}, {"type": "BASIC"}], "default": {"type": "PREMIUM"}},
        "kind": {"const": {"type": "FIXED"}, "examples": [{"type": "SAMPLE"}]},
        "default": {"type": "string"}, "enum": {"type": "array", "items": {"type": ["string", "null"]}},
        "type": {"type": "integer"}}}


def test_one_unresolvable_annotation_leaves_the_other_parameters_typed():
    # "Decimal" is not defined here, as with an import that exists only under TYPE_CHECKING.
    def pay(amount: "int", currency: "Literal['USD', 'KRW']", price: "Decimal" = None,  # noqa: F821
            unit: "Unit" = Unit.CELSIUS):
        return unit

    assert dmcall.schema_of(pay)["function"]["parameters"]["properties"] == {
        "amount": {"type": "integer"}, "currency": {"type": "string", "enum": ["USD", "KRW"]},
        "price": {"default": None},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}
    assert dmcall.invoke(pay, {"amount": 1, "currency": "USD", "unit": "fahrenheit"}) is Unit.FAHRENHEIT


@pytest.mark.parametrize("bad, error", [
    (lambda city: city, ValueError),
    ("get_weather", TypeError),
    ({"name": "", "parameters": {"type": "object"}}, ValueError),
    ({"name": "x", "parameters": {"type": "array"}}, ValueError),
    ({"name": "x", "parameters": {"type": "object", "properties": {"a": {"type": "nonsense"}}}}, ValueError),
    ({"name": "x", "parameters": {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}}}, ValueError),
    ({"name": "x", "parameters": {"type": "object", "properties": {"a": {"$ref": "https://example.com/a.json"}}}},
     ValueError),
    ({"name": "x", "parameters": {"type": "object", "properties": {"a": {"$ref": "#/properties/a"}}}}, ValueError),
], ids=["unnamed-lambda", "string", "no-name", "not-object", "invalid-schema", "missing-ref", "remote-ref", "self-ref"])
def test_bad_tools(bad, error):
    with pytest.raises(error):
        dmcall.toolset([bad])
    if isinstance(bad, dict) and "$ref" in str(bad):  # the client refuses it too, before any request
        with pytest.raises(ValueError, match=r"\$ref"):
            dmcall.DMCall(ScriptedModel(), decision_model=ScriptedJev().jev, tools=[bad])


def test_unsupported_signatures():
    def positional(city, /):
        pass

    def odd(value: complex):
        pass

    with pytest.raises(TypeError, match="positional-only"):
        dmcall.schema_of(positional)
    with pytest.raises(TypeError, match="parameters="):
        dmcall.schema_of(odd)
    with pytest.raises(ValueError, match="Two tools"):
        dmcall.toolset([WEATHER, WEATHER])
    with pytest.raises(ValueError, match="not one of the tools"):
        dmcall.toolset([WEATHER], functions={"nope": print})

    async def remote(city: str):
        return city

    with pytest.raises(TypeError, match="Async"):
        dmcall.invoke(remote, {"city": "Seoul"})


def test_calls_from_every_sdk_shape():
    seoul = [{"name": "get_weather", "arguments": {"city": "Seoul"}}]
    shapes = [
        {"name": "get_weather", "arguments": {"city": "Seoul"}},
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Seoul"}'},
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}},
        {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}},
    ]
    for shape in shapes:
        assert dmcall.normalize_calls([shape]) == seoul

    class SDKCall:
        def model_dump(self, mode="json"):
            return shapes[1]

    assert dmcall.normalize_calls([SDKCall()]) == seoul
    assert dmcall.normalize_calls(None) == []  # SDKs report no call as tool_calls=None
    assert dmcall.normalize_calls({"name": "f", "arguments": "{oops"}) == [{"name": "f", "arguments": None}]
    with pytest.raises(TypeError):
        dmcall.normalize_calls(3)
