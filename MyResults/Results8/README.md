1. I think I fixed the issue with the unexpected things happened when agent won()
2. Also seems that when ms_per_tick is low agent is not observing the environment as expected. So I increased it to 10
3. Here we had very good results because If agent won I gave him more life at the start.
So I want to see if I remove it that we don't get worst results.