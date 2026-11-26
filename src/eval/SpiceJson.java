package edu.anu.spice;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

class SpiceJson {
    public static void main(String[] args) throws Exception {
        var arguments = SpiceArguments.class.getDeclaredConstructor(String[].class);
        arguments.setAccessible(true);
        var factory = SpiceScorer.class.getDeclaredConstructor();
        factory.setAccessible(true);
        SpiceArguments options = arguments.newInstance((Object) Arrays.copyOfRange(args, 1, args.length));
        SpiceScorer scorer = factory.newInstance();
        scorer.scoreBatch(options);
        // SPICE's CLI uses Nashorn only to pretty-print this JSON.
        Files.write(Paths.get(args[0]), scorer.stats.toJSONString().getBytes(StandardCharsets.UTF_8));
    }
}
