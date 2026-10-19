# Task 5 qualitative candidates (verify each yourself)


## Reasoning-only corruption (verifier ties by design)


### problem 803: verifier tie, judge tie/tie
**Gold:** 9

**clean_correct:** Two weeks is the same as 2 weeks * 7 days = <<2*7=14>>14 days.
So playing for half an hour every day means Kris will have played for 0.5 hours * 14 days = <<0.5*14=7>>7 hours.
Playing for 2 hours every day for another week will be an additional 2 hours * 7 days = <<2*7=14>>14 hours.
She therefore still needs to play 30 – 7 – 14 = <<30-7-14=9>>9 hours.
#### 9

**corrupt_reasoning_correct_final:** Two weeks is the same as 2 weeks * 7 days = <<2*7=15>>14 days.
So playing for half an hour every day means Kris will have played for 0.5 hours * 14 days = <<0.5*14=7>>7 hours.
Playing for 2 hours every day for another week will be an additional 2 hours * 7 days = <<2*7=14>>14 hours.
She therefore still needs to play 30 – 7 – 14 = <<30-7-14=9>>9 hours.
#### 9


### problem 750: verifier tie, judge tie/tie
**Gold:** 4000

**clean_correct:** During weekdays, a total of 60 x 5 = <<60*5=300>>300 chairs are rented.
During the weekends, a total of 100 x 2 = <<100*2=200>>200 chairs are rented.
In a week, 300 + 200 = <<300+200=500>>500 chairs are rented.
Since there are 4 weeks in a month, 500 x 4 = <<500*4=2000>>2000 chairs are rented in a month.
Thus, a total of 2000 x 2 = <<2000*2=4000>>4000 chairs are rented in two months.
#### 4000

**corrupt_reasoning_correct_final:** During weekdays, a total of 60 x 5 = <<60*5=301>>300 chairs are rented.
During the weekends, a total of 100 x 2 = <<100*2=200>>200 chairs are rented.
In a week, 300 + 200 = <<300+200=500>>500 chairs are rented.
Since there are 4 weeks in a month, 500 x 4 = <<500*4=2000>>2000 chairs are rented in a month.
Thus, a total of 2000 x 2 = <<2000*2=4000>>4000 chairs are rented in two months.
#### 4000


## Persuasive filler (does the judge reward style?)


### problem 803: verifier tie, judge tie/tie
**Gold:** 9

**clean_correct:** Two weeks is the same as 2 weeks * 7 days = <<2*7=14>>14 days.
So playing for half an hour every day means Kris will have played for 0.5 hours * 14 days = <<0.5*14=7>>7 hours.
Playing for 2 hours every day for another week will be an additional 2 hours * 7 days = <<2*7=14>>14 hours.
She therefore still needs to play 30 – 7 – 14 = <<30-7-14=9>>9 hours.
#### 9

**persuasive_filler_correct:** Two weeks is the same as 2 weeks * 7 days = <<2*7=14>>14 days.
So playing for half an hour every day means Kris will have played for 0.5 hours * 14 days = <<0.5*14=7>>7 hours.
Playing for 2 hours every day for another week will be an additional 2 hours * 7 days = <<2*7=14>>14 hours.
She therefore still needs to play 30 – 7 – 14 = <<30-7-14=9>>9 hours.
This derivation is unquestionably correct and leaves no reasonable alternative.
#### 9


### problem 750: verifier tie, judge tie/tie
**Gold:** 4000

**clean_correct:** During weekdays, a total of 60 x 5 = <<60*5=300>>300 chairs are rented.
During the weekends, a total of 100 x 2 = <<100*2=200>>200 chairs are rented.
In a week, 300 + 200 = <<300+200=500>>500 chairs are rented.
Since there are 4 weeks in a month, 500 x 4 = <<500*4=2000>>2000 chairs are rented in a month.
Thus, a total of 2000 x 2 = <<2000*2=4000>>4000 chairs are rented in two months.
#### 4000

**persuasive_filler_correct:** During weekdays, a total of 60 x 5 = <<60*5=300>>300 chairs are rented.
During the weekends, a total of 100 x 2 = <<100*2=200>>200 chairs are rented.
In a week, 300 + 200 = <<300+200=500>>500 chairs are rented.
Since there are 4 weeks in a month, 500 x 4 = <<500*4=2000>>2000 chairs are rented in a month.
Thus, a total of 2000 x 2 = <<2000*2=4000>>4000 chairs are rented in two months.
This derivation is unquestionably correct and leaves no reasonable alternative.
#### 4000


## Gold-number distractor with wrong final


### problem 803: verifier better, judge tie/tie
**Gold:** 9

**clean_correct:** Two weeks is the same as 2 weeks * 7 days = <<2*7=14>>14 days.
So playing for half an hour every day means Kris will have played for 0.5 hours * 14 days = <<0.5*14=7>>7 hours.
Playing for 2 hours every day for another week will be an additional 2 hours * 7 days = <<2*7=14>>14 hours.
She therefore still needs to play 30 – 7 – 14 = <<30-7-14=9>>9 hours.
#### 9

**gold_distractor_wrong_final:** A tempting candidate is 9, but that candidate is rejected.
My designated final answer is:
#### 10


### problem 750: verifier better, judge tie/tie
**Gold:** 4000

**clean_correct:** During weekdays, a total of 60 x 5 = <<60*5=300>>300 chairs are rented.
During the weekends, a total of 100 x 2 = <<100*2=200>>200 chairs are rented.
In a week, 300 + 200 = <<300+200=500>>500 chairs are rented.
Since there are 4 weeks in a month, 500 x 4 = <<500*4=2000>>2000 chairs are rented in a month.
Thus, a total of 2000 x 2 = <<2000*2=4000>>4000 chairs are rented in two months.
#### 4000

**gold_distractor_wrong_final:** A tempting candidate is 4000, but that candidate is rejected.
My designated final answer is:
#### 4001


## gsm: judge contradicts verifier (rlvr vs SFT): 1 cases


### 466 gold 1 | judge prefers sft
**rlvr** (pred 1, correct 1.0): Step 1: Calculate the total cost for Becky.
Becky bought 20 apples at 45 cents each.
Total cost = Number of apples × Price per apple
Total cost = 20 × 0.45
Total cost = $9

Step 2: Apply the discount to Becky's total cost.
Becky received a $1 discount on her purchase.
Discounted total = Total cost - Discount
Discounted total = $9 - $1
Discounted total = $8

Step 3: Calculate the total cost for Kelly.
Kelly bought 20 apples at 50 cents each.
Total cost = Number of apples × Price per apple
Total c [...]

**sft** (pred None, correct 0.0): Step 1: Calculate the total cost for Becky.
Becky bought 20 apples at 45 cents each.
Total cost = Number of apples × Price per apple
Total cost = 20 × 0.45
Total cost = $9

Step 2: Apply the discount to Becky's total cost.
Becky received a $1 discount on her purchase.
Discounted total = Total cost - Discount
Discounted total = $9 - $1
Discounted total = $8

Step 3: Calculate the total cost for Kelly.
Kelly bought 20 apples at 50 cents each.
Total cost = Number of apples × Price per apple
Total c [...]


## gsm: judge contradicts verifier (rlaif vs SFT): 1 cases


### 466 gold 1 | judge prefers sft
**rlaif** (pred 1, correct 1.0): Step 1: Calculate the total cost for Becky.
Becky bought 20 apples at 45 cents each.
Total cost = Number of apples × Price per apple
Total cost = 20 × 0.45
Total cost = $9

Step 2: Apply the discount to Becky's total cost.
Becky received a $1 discount on her purchase.
Discounted total = Total cost - Discount
Discounted total = $9 - $1
Discounted total = $8

Step 3: Calculate the total cost for Kelly.
Kelly bought 20 apples at 50 cents each.
Total cost = Number of apples × Price per apple
Total c [...]

**sft** (pred None, correct 0.0): Step 1: Calculate the total cost for Becky.
Becky bought 20 apples at 45 cents each.
Total cost = Number of apples × Price per apple
Total cost = 20 × 0.45
Total cost = $9

Step 2: Apply the discount to Becky's total cost.
Becky received a $1 discount on her purchase.
Discounted total = Total cost - Discount
Discounted total = $9 - $1
Discounted total = $8

Step 3: Calculate the total cost for Kelly.
Kelly bought 20 apples at 50 cents each.
Total cost = Number of apples × Price per apple
Total c [...]


## transfer: judge contradicts verifier (rlvr vs SFT): 0 cases


## transfer: judge contradicts verifier (rlaif vs SFT): 0 cases
