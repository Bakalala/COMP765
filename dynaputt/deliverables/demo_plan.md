# Dyna Putt station demo plan

The session is a seated station with a small group: 1-2 slides or videos plus Q&A.
Use `deliverables/station_slides.pdf` and the live browser demo. The manual launch
check on the Mac has not been done yet; see the checklist.

## 60-90 second opening

"I asked whether an agent can learn a mini-golf course with fewer real shots if it
also practices in a model of the course. The agent sees the cell the ball stopped in
and picks a direction and power. Ordinary Q-learning updates once per real shot.
Dyna-Q also remembers what each shot did and replays those memories for extra updates.
At first the answer looked simple: more practice, fewer real shots. But when I
checked, a lot of that benefit came from how the agent's values started. Every shot
costs a point, so starting at zero makes everything untried look great. Planning was
partly just cleaning up that optimism. If you remove the optimism entirely, though, the
agent often never finds the hole on long courses. So how much planning helps depends
on where the values start."

## Live sequence

1. Open the Wall-gate course and point out the start, hole, wall, and bumper. Say the
   state is just the cell where the ball stops.
2. Pick a direction and power and press **Take shot** (or keys `1`-`8`, `P`, `Space`).
   Point to the path and stroke counter.
3. Press **Reset course**, then press **Let Dyna-Q shoot** repeatedly. It should sink in
   5 shots. Switch to the easy course in the menu; it should sink in 2.
4. Open the slides. Show the planning results, then the initial-value result.

Say that the in-browser agent is illustrative: it trains for 700 episodes with one
random stream and is not one of the measured runs.

## Plain-language planning explanation

"After a real shot, Dyna-Q updates its value for that shot and remembers what happened.
Then it picks earlier remembered shots at random and updates them again as if they had
just happened. That reuses experience. It can't tell the agent about a shot it has
never tried, so it can't replace exploring."

## Likely questions

**Do more planning updates always help?**
On average in these courses, yes, but with sharply diminishing returns. One update per
shot gives over half of the gain up to 50 updates. An earlier five-seed run even showed
5 beating 20 on the easy course; with 100 seeds that turned out to be noise, not a real
reversal or an effect of the random-number setup.

**So is Dyna-Q better than Q-learning?**
With the original zero start, it needs about 3.2 times fewer shots on Wall-gate. With a
non-optimistic start, the threshold advantage disappears on the two-shot course and is
about 1.3 to 1.5 times on Wall-gate. Plain Q-learning with a sensible start is competitive.

**Why not always start pessimistic?**
Because then only random exploration finds the hole. At the lowest starting value, only
6 of 100 Q-learning runs and 32 of 100 Dyna-Q runs reach the threshold on the 8-shot
course. All 100 succeed when starting at 0 or -5.

**Is this a realistic simulator or a modern world model?**
No. It is a small deterministic grid with an exact table, which is what makes it
inspectable. It says nothing about model error, which is the hard part for methods
such as PETS, TD-MPC, or DreamerV3.

## Mac checklist (student)

- Start with `python play.py`, or open `index.html` directly. It needs no packages.
- Confirm the page opens at 0 / 20 strokes with the ball visible at START.
- Run through the live sequence once, including Reset and switching course.
- Open `deliverables/station_slides.pdf` and check it at the display size you will use.
- Practice once on battery.

## Fallback

If the browser demo fails, use the slides. They contain the course maps and results,
and you can walk through one transition by hand: direction, collision, reward, and what
gets stored. The evidence is the repeated experiments; the demo only makes it tangible.
