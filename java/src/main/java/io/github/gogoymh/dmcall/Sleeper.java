package io.github.gogoymh.dmcall;

/** Waits between HTTP retries. Tests pass a sleeper that returns at once. */
@FunctionalInterface
public interface Sleeper {
    /** Sleeps with {@link Thread#sleep}. */
    Sleeper SYSTEM = seconds -> Thread.sleep(Math.max(0L, Math.round(seconds * 1000.0)));

    void sleep(double seconds) throws InterruptedException;
}
