# Failure Analysis (text -> image)

250 of 5000 caption queries ranked their true image worse than 100 (mean rank 27.5, median 5).

## A. Worst-ranked captions (semantic vs. visual mismatch)

The true image is what the caption describes; the top results are what CLIP thought matched better. Mismatches show CLIP anchoring on dominant visual concepts and under-weighting the detail the caption singled out.

**"A photo of an outside with various things in the scene."**  (gold rank 3880)
- gold image: val2017/000000229747.jpg — categories: airplane
- returned 0.286: val2017/000000246308.jpg — chair, tv, laptop, mouse, keyboard
- returned 0.285: val2017/000000255536.jpg — person, motorcycle, umbrella, chair
- returned 0.284: val2017/000000491216.jpg — cat, bottle, spoon, banana, chair, potted plant, oven, sink, refrigerator

**"A european city in nice a sunny bright day"**  (gold rank 2690)
- gold image: val2017/000000095069.jpg — categories: clock
- returned 0.259: val2017/000000304396.jpg — clock
- returned 0.258: val2017/000000376625.jpg — person, car, motorcycle, train, bottle
- returned 0.251: val2017/000000209222.jpg — person, car, truck, bench, handbag

**"A thing is in the outline and it shows up like something"**  (gold rank 2254)
- gold image: val2017/000000360393.jpg — categories: bowl, pizza
- returned 0.253: val2017/000000404191.jpg — refrigerator
- returned 0.244: val2017/000000024610.jpg — backpack, bottle, chair, couch, laptop, book
- returned 0.244: val2017/000000163562.jpg — person, frisbee

**"An individual is capture in the stillness of the picture."**  (gold rank 2158)
- gold image: val2017/000000425361.jpg — categories: person, wine glass, fork, knife, bowl, pizza, chair, laptop, mouse
- returned 0.266: val2017/000000456496.jpg — person, bird, handbag
- returned 0.262: val2017/000000384651.jpg — cup, bed
- returned 0.261: val2017/000000082765.jpg — bed, laptop

**"A being is doing something as of right now that is splendid."**  (gold rank 2136)
- gold image: val2017/000000425227.jpg — categories: person, kite, surfboard
- returned 0.243: val2017/000000112798.jpg — cat, laptop, book
- returned 0.242: val2017/000000189806.jpg — cat, dog, couch
- returned 0.242: val2017/000000565962.jpg — person, cat

**"Their is a hadron and their  in this lot,"**  (gold rank 2006)
- gold image: val2017/000000087875.jpg — categories: fire hydrant
- returned 0.253: val2017/000000453341.jpg — couch, tv, mouse, keyboard, cell phone, teddy bear
- returned 0.250: val2017/000000573943.jpg — truck
- returned 0.250: val2017/000000536947.jpg — truck, bottle, chair, refrigerator

**"A couple of men standing next to each other."**  (gold rank 1889)
- gold image: val2017/000000415727.jpg — categories: person, bench, backpack, baseball bat
- returned 0.277: val2017/000000212573.jpg — person, car, traffic light, umbrella
- returned 0.260: val2017/000000495732.jpg — person, couch, remote
- returned 0.257: val2017/000000069213.jpg — person, motorcycle

**"A photograph of an outside with numerous things in the scene."**  (gold rank 1677)
- gold image: val2017/000000336658.jpg — categories: person, car, bus
- returned 0.300: val2017/000000376264.jpg — bottle, cup, knife, laptop, cell phone, book
- returned 0.296: val2017/000000564091.jpg — person, handbag, bottle, cell phone
- returned 0.295: val2017/000000348481.jpg — laptop, mouse, remote, cell phone, book

## B. Compositional probes

Top-3 images CLIP returns for queries that require relations, counting, or negation. Inspect whether the results actually honor the constraint (they usually don't).

### spatial relation

**"a cat to the left of a dog"**
- 0.305: val2017/000000219578.jpg — cat, dog, couch
- 0.295: val2017/000000241326.jpg — cat, dog, couch
- 0.293: val2017/000000071226.jpg — cat, dog, bed, book

**"a person standing behind a car"**
- 0.291: val2017/000000466156.jpg — car, cat
- 0.288: val2017/000000097679.jpg — person, car, snowboard
- 0.287: val2017/000000010363.jpg — bicycle, car, cat, bottle

**"a bottle on top of a book"**
- 0.276: val2017/000000102707.jpg — bottle, cup, oven
- 0.272: val2017/000000542776.jpg — person, bird, bed, book
- 0.269: val2017/000000551439.jpg — person, bed, book

### counting

**"exactly three dogs"**
- 0.287: val2017/000000318238.jpg — cat, dog, bed
- 0.286: val2017/000000236784.jpg — dog, couch
- 0.284: val2017/000000447200.jpg — dog

**"two people and one bicycle"**
- 0.298: val2017/000000038829.jpg — person, bicycle, motorcycle, backpack
- 0.281: val2017/000000414510.jpg — person, bicycle, car, truck, backpack, handbag, cell phone
- 0.280: val2017/000000289343.jpg — person, bicycle, bench, dog

### negation

**"a street with no cars"**
- 0.300: val2017/000000314251.jpg — person, car, motorcycle
- 0.283: val2017/000000115946.jpg — person, car, traffic light, potted plant
- 0.281: val2017/000000476491.jpg — (none)

**"a plate with food but no fork"**
- 0.304: val2017/000000561889.jpg — fork, knife, broccoli, carrot
- 0.285: val2017/000000013004.jpg — banana, dining table
- 0.279: val2017/000000419312.jpg — cup, fork, knife, bowl, carrot, cake, dining table

### attribute binding (swap test)

If binding were understood, swapping the colors would change the results. Cosine near 1.0 and Jaccard near 1.0 mean CLIP treats the swapped queries as essentially the same — a binding failure.

| Query A | Query B | query cosine | top-10 Jaccard |
|---|---|---|---|
| a red car and a blue bus | a blue car and a red bus | 0.978 | 0.67 |
| a white dog and a black cat | a black dog and a white cat | 0.988 | 0.67 |
| a man in a red shirt and a woman in a green shirt | a man in a green shirt and a woman in a red shirt | 0.988 | 0.54 |
