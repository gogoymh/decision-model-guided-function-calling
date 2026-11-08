package io.github.gogoymh.dmcall;

/** A tool handler failed with a checked exception; the cause holds it. */
public class ToolExecutionException extends RuntimeException {
    private final String tool;

    public ToolExecutionException(String tool, Throwable cause) {
        super("The handler of tool " + Py.strRepr(String.valueOf(tool)) + " failed.", cause);
        this.tool = tool;
    }

    /** The tool whose handler failed. */
    public String tool() {
        return tool;
    }
}
