package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The outcome of one DM-Call request.
 *
 * <p>{@link #calls()} is the final candidate: {@code []} is a decision to make no call and
 * {@code null} an unreadable candidate (cut off, filtered or refused). {@link #accepted()} says
 * whether it passed every check; it is {@code null} for an initial no call returned without
 * verification ({@link #initialNoCallPreserved()}). When no candidate passed within the repair
 * limit ({@link #exhausted()}), the final candidate is the first one (the paper's rule) or the last
 * one; {@link #finalAttempt()} is its index in {@link #trace()}, which keeps every candidate,
 * judgment and feedback record in order.
 */
public final class Result implements JsonConvertible {
    private final List<Call> calls;
    private final Boolean accepted;
    private final boolean exhausted;
    private final int repairs;
    private final String text;
    private final boolean executed;
    private final List<Object> outputs;
    private final double latencySeconds;
    private final List<TraceStep> trace;
    private final Integer finalAttempt;
    private final boolean initialNoCallPreserved;

    Result(List<Call> calls, Boolean accepted, boolean exhausted, int repairs, String text, boolean executed,
           List<Object> outputs, double latencySeconds, List<TraceStep> trace, Integer finalAttempt,
           boolean initialNoCallPreserved) {
        this.calls = Calls.copy(calls);
        this.accepted = accepted;
        this.exhausted = exhausted;
        this.repairs = repairs;
        this.text = text;
        this.executed = executed;
        this.outputs = Collections.unmodifiableList(new ArrayList<>(outputs));
        this.latencySeconds = latencySeconds;
        this.trace = List.copyOf(trace);
        this.finalAttempt = finalAttempt;
        this.initialNoCallPreserved = initialNoCallPreserved;
    }

    /** The final candidate: {@code []} for no call, {@code null} if unreadable. */
    public List<Call> calls() {
        return calls;
    }

    /** Whether the final candidate passed every check; {@code null} for an unverified initial no call. */
    public Boolean accepted() {
        return accepted;
    }

    /** No candidate passed within the repair limit. */
    public boolean exhausted() {
        return exhausted;
    }

    /** Repairs made: candidates after the first. */
    public int repairs() {
        return repairs;
    }

    /** The final candidate's text, or {@code null}. */
    public String text() {
        return text;
    }

    /** Whether {@link DMCall#run} executed the final call. */
    public boolean executed() {
        return executed;
    }

    /** What the executed handlers returned, in call order. */
    public List<Object> outputs() {
        return outputs;
    }

    public double latencySeconds() {
        return latencySeconds;
    }

    /** Every round in order. */
    public List<TraceStep> trace() {
        return trace;
    }

    /** The index of the final candidate in {@link #trace()}. */
    public Integer finalAttempt() {
        return finalAttempt;
    }

    /** The first candidate was a readable no call, returned without verification or repair. */
    public boolean initialNoCallPreserved() {
        return initialNoCallPreserved;
    }

    /** The first call of the final candidate, or {@code null} for no call or an unreadable candidate. */
    public Call call() {
        return calls == null || calls.isEmpty() ? null : calls.get(0);
    }

    /** The final candidate is a decision to make no call. */
    public boolean noCall() {
        return calls != null && calls.isEmpty();
    }

    /** The final candidate is unreadable. */
    public boolean unreadable() {
        return calls == null;
    }

    /** What the executed handler returned (a list if several calls ran), or {@code null}. */
    public Object output() {
        if (outputs.isEmpty()) {
            return null;
        }
        return outputs.size() == 1 ? outputs.get(0) : outputs;
    }

    /** Each candidate with its verdict, scores and schema errors, oldest first. */
    public List<Attempt> attempts() {
        List<Attempt> rounds = new ArrayList<>();
        for (TraceStep step : trace) {
            Judgment check = step.verification();
            rounds.add(new Attempt(step.proposal().calls(), step.gatePass(),
                    check == null ? null : check.functionProbability(),
                    check == null ? null : check.argumentsProbability(),
                    step.schema() == null ? List.of() : step.schema().errors()));
        }
        return rounds;
    }

    /** The decision model's probabilities of yes for the final candidate ({@code null} if not asked). */
    public Probabilities probabilities() {
        if (trace.isEmpty()) {
            return new Probabilities(null, null);
        }
        int index = finalAttempt == null ? trace.size() - 1 : finalAttempt;
        Judgment check = trace.get(index).verification();
        return check == null ? new Probabilities(null, null)
                : new Probabilities(check.functionProbability(), check.argumentsProbability());
    }

    /** Token counts summed over every generator request and every decision-model request. */
    public Usage usage() {
        Map<String, Long> generator = new LinkedHashMap<>();
        Map<String, Long> decisionModel = new LinkedHashMap<>();
        for (TraceStep step : trace) {
            add(generator, step.proposal().usage());
            if (step.verification() != null) {
                add(decisionModel, step.verification().usage());
            }
        }
        return new Usage(generator, decisionModel);
    }

    private static void add(Map<String, Long> total, Map<String, Long> counts) {
        if (counts == null) {
            return;
        }
        counts.forEach((key, value) -> {
            if (value != null) {
                total.merge(key, value, Long::sum);
            }
        });
    }

    /**
     * A summary: calls, accepted, exhausted, repairs, final_attempt, text,
     * initial_no_call_preserved, attempts, probabilities and usage.
     */
    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.set("calls", Calls.toJson(calls));
        node.put("accepted", accepted);
        node.put("exhausted", exhausted);
        node.put("repairs", repairs);
        node.put("final_attempt", finalAttempt);
        node.put("text", text);
        node.put("initial_no_call_preserved", initialNoCallPreserved);
        ArrayNode rounds = node.putArray("attempts");
        attempts().forEach(attempt -> rounds.add(attempt.toJson()));
        node.set("probabilities", probabilities().toJson());
        node.set("usage", usage().toJson());
        return node;
    }

    @Override
    public String toString() {
        String decision;
        if (calls == null) {
            decision = "unreadable candidate";
        } else if (calls.isEmpty()) {
            decision = "no call";
        } else {
            List<String> parts = new ArrayList<>();
            calls.forEach(call -> parts.add(call.toString()));
            decision = "call=" + String.join("; ", parts);
        }
        String result = "Result(" + decision + ", accepted=" + accepted + ", repairs=" + repairs;
        if (executed) {
            result += ", output=" + output();
        }
        return result + ")";
    }
}
