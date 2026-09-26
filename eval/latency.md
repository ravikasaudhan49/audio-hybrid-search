Latency: 49 golden queries × 10 rounds, concurrency 5, summary answer off (warm caches).

| end-to-end (client) | p50 | p90 | p99 | mean | max | throughput | errors |
|---|---|---|---|---|---|---|---|
| all requests (n=490) | 186 ms | 274 ms | 371 ms | 194 ms | 497 ms | 24.9 req/s | 0 |

| server stage | p50 | p90 | p99 |
|---|---|---|---|
| understand | 0.0 ms | 0.0 ms | 0.0 ms |
| embed | 0.1 ms | 0.1 ms | 0.3 ms |
| retrieve | 122.7 ms | 193.9 ms | 274.4 ms |
| rerank | 2.2 ms | 3.2 ms | 4.9 ms |
| post | 0.3 ms | 0.6 ms | 1.1 ms |
| total | 127.5 ms | 197.7 ms | 279.2 ms |

| query | p50 | p90 | p99 | reranked | results |
|---|---|---|---|---|---|
| n01 Mars helicopter flights | 141 | 258 | 280 | yes | 0 |
| n02 recipe for chocolate chip cookies | 98 | 116 | 125 | yes | 0 |
| n03 James Webb Space Telescope mirror alignment | 130 | 172 | 187 | yes | 0 |
| n04 SpaceX Starship heat shield tiles | 116 | 199 | 204 | yes | 0 |
| q01 registry not large enough | 110 | 225 | 252 | yes | 1 |
| q02 why is it hard to find stem cell matches for minority patien | 142 | 199 | 231 | yes | 1 |
| q03 cytokines | 257 | 326 | 340 | yes | 2 |
| q04 why can't you simply multiply stem cells in a lab dish | 238 | 346 | 372 | yes | 3 |
| q05 what would people look like if we evolved on a much heavier  | 223 | 322 | 333 | yes | 1 |
| q06 "affect gene expression" | 183 | 269 | 361 | yes | 2 |
| q07 the core question of the space station stem cell experiment | 184 | 315 | 418 | yes | 4 |
| q08 microgravity expanded stem cells study | 190 | 338 | 399 | yes | 6 |
| q09 growing cells without them maturing into specialized cells | 179 | 331 | 360 | yes | 4 |
| q10 what did Zubair say about differentiation | 230 | 337 | 368 | yes | 4 |
| q11 what did the host ask about the childhood dream in Nigeria | 223 | 310 | 333 | yes | 1 |
| q12 Larry Harvey | 195 | 227 | 285 | yes | 1 |
| q13 how did the researcher first get connected to space research | 160 | 219 | 290 | yes | 10 |
| q14 strok recovry | 165 | 239 | 270 | yes | 4 |
| q15 success comes down to knowing the right people | 187 | 231 | 234 | yes | 1 |
| q16 Harry Carroll | 191 | 244 | 266 | yes | 2 |
| q17 "we will leave as a team" | 221 | 242 | 255 | yes | 3 |
| q18 what did the flight director promise his controllers before  | 171 | 233 | 263 | yes | 4 |
| q19 Gemini eight | 205 | 271 | 288 | yes | 1 |
| q20 "risk is the price of progress" | 201 | 277 | 288 | yes | 1 |
| q21 why he enjoyed decisions having no grey areas | 187 | 247 | 257 | yes | 2 |
| q22 crew chief hand salute | 196 | 245 | 270 | yes | 1 |
| q23 what did the host ask about tough and competent | 259 | 300 | 344 | yes | 3 |
| q24 what did Kranz say about accountability | 225 | 264 | 339 | yes | 4 |
| q25 training program for new generations of shuttle flight contr | 142 | 184 | 226 | yes | 1 |
| q26 special forces in Afghanistan | 153 | 207 | 224 | yes | 1 |
| q27 "failure is not an option" | 188 | 240 | 251 | yes | 4 |
| q28 how did the famous Apollo 13 movie line come about | 150 | 161 | 172 | yes | 4 |
| q29 Jery Bostik | 159 | 210 | 215 | yes | 3 |
| q30 Jim Lovell book | 199 | 229 | 240 | yes | 1 |
| q31 what he actually told the team when the crew's return was in | 173 | 199 | 200 | yes | 4 |
| q32 what did the guest say about trust | 235 | 283 | 317 | yes | 6 |
| q33 does failure mean never making mistakes or refusing to quit | 162 | 211 | 281 | yes | 3 |
| q34 Center for the Advancement of Science in Space | 238 | 288 | 333 | yes | 9 |
| q35 when did the nonprofit begin running the orbiting laboratory | 174 | 220 | 290 | yes | 1 |
| q36 Department of Energy national labs | 203 | 383 | 485 | yes | 7 |
| q37 why is sending an experiment to orbit so difficult for a sci | 178 | 208 | 274 | yes | 10 |
| q38 Tetris | 172 | 223 | 223 | yes | 1 |
| q39 "astronaut proof it" | 170 | 216 | 241 | yes | 1 |
| q40 who pays for transporting research to and from the station | 241 | 312 | 325 | yes | 9 |
| q41 what did Gary Jordan say about astronauts running experiment | 220 | 301 | 305 | yes | 7 |
| q42 Merck and Eli Lilly | 155 | 233 | 294 | yes | 1 |
| q43 protein crystallization | 184 | 272 | 307 | yes | 1 |
| q44 Bristol Myers Squib | 170 | 207 | 224 | yes | 2 |
| q45 what did the host say about being the new kid on the block | 193 | 265 | 306 | yes | 1 |
