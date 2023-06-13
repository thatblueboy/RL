1. Are current reward and next state return independant? If no, how is matrix form of bellman equation derived?

2. How is the bellman equation for action value function, $q(s_t, a_t)_{\pi} = \mathbb{E}_{\pi} [R_{t+1} + \gamma q(S_{t+1}, A_{t+1})|s, a]$ derived? we don't know $a_{t+1}$ and thus $q(s_{t+1}, a_{t+1})$ ? shouldnt it be $q(s, a)_{\pi} = \mathbb{E}_{\pi} [R_{t+1} + \gamma V(S_{t+1})|s_t, a_t]$ ?
<br>Ans: in the case of bellman equation, we dont know the next state, yet we mention $G_{t+1}$ in the equation. Note that this is the expectation and we take weighted sum over all possible states. Same thing is done for action value function. We take weighted sum over all possible actions and states

3. The definition for optimal action value function is $q_*(s, a) = \max_{\pi} q_{\pi}(s, a)$. Is this for all actions given a state? wouldnt this be true for only certain correct actions?

4. There is always a deterministic optimal policy for any MDP. Is this true? If yes, how is it proved? What about stochastic processes like stone paper scissor, or gridworld from that research paper?

5. OpenAI says Agents should really only reinforce actions on the basis of their consequences. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come after. I dont get it.

