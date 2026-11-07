# Station script

Bring: laptop with `demo/index.html` open, `station_slides.pdf`, and
`putt_world_clip.mp4` as a backup.

## 60-second opening

"World models let an agent imagine actions before taking them. I rebuilt a recent one,
LeWorldModel, for mini-golf. It only ever saw tiny pictures of random shots, with no
rules, score, or wall positions. Recent papers say this kind of model ignores walls,
and I expected that too. It didn't. Its sense of distance jumps right at the walls.
So I asked why, and drew a fake wall that the ball rolls straight through. That left no
trace at all. The model doesn't learn walls by looking at them. It learns them because
shots get blocked by them, and bouncing makes the effect about twice as strong."

## Live sequence

1. Open **Gate**. Drag back from the ball and take a shot.
2. Press **Reset**, then **Let the world model plan**. Point to the faint imagined shots.
3. Turn on **Mind view**. Point out how the colour changes sharply at the wall.
4. Open slide 1 (the maps), then slide 2 (ghost wall and shot power).
5. Show **Pocket** honestly: the agent often fails here.

The demo agent sank 10 of 10 on Gate and Slalom, and on Open with the shot graph.
On Pocket it sank 1 of 10 with straight-line distance and 5 of 10 with the shot graph,
so expect failures there and say so.

## Likely questions

**How does it know where walls are?** It doesn't get told. Points on opposite sides of a
wall have very different futures, because shots get blocked, so the predictor has to
separate them. A wall that blocks without bouncing gives a gap of 3.82, against 7.88 for
a normal wall and 0.96 for a ghost wall.

**Why did other papers find the opposite?** I can't be sure. Their actions are small
steps, and when I weakened shots here, the wall effect shrank from 7.88 to 1.93. That is
a hypothesis, because I didn't run their benchmark.

**Did your improvement work?** Not clearly. A Dyna-style graph over latent states didn't
beat plain latent distance, and it was worse on the open course until I added a fix.
Pocket is still unsolved, but that's mainly the predictor: with true physics, the same
latent distance sinks 100%.

**How big is this?** 1.9M parameters, about five minutes per model on a laptop, compared
with about 15M parameters in the original.

**How solid is it?** Three seeds, four hand-built courses, a small simulator. The ghost
wall effect is large, but the graph differences are within seed noise, and the wall
studies were designed after seeing the first results.

## If the demo fails

Play `putt_world_clip.mp4` and use the slides. The evidence is in the experiments, and
the demo just makes it tangible.
