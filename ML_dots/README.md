<h1>ML Dots</h1>
<hr/>
<ul>
  <li>
    This is a program that uses the genetic algorithm and is one of the very first ML algorithms I have ever written (The genetic algorithm is actually considered a "heuristic" algorithm, but i mean the definition of "heuristic" means self learning so)
  </li>
  <li>
    Things you can play with is in the global variables of 'showcase.py'
    <ul>
      <li>
        Course number
      </li>
      <li>
        Population size
      </li>
      <li>
        Mutation Rate - Percentage chance of move deviance after each mutation
      </li>
      <li>
        Starting Moves - Number of moves the dots are allowed before next mutation
      </li>
      <li>
        Move Increments - Number of moves increase after each mutation
      </li>
      <li>
        Generation Increment - Number of Generations that must pass before each mutation
      </li>
  </ul>  
  </li>
</ul>
<h3>Explanation</h3>
<ul>
  <li>
    Each dot is given a list of random moves
  </li>
  <li>
    They all then execute these moves and the best performing one, which is based on some reward function that i wrote, is the "Parent" for the next generation
  </li>
  <li>
    The Parent then makes Babies and each baby has the same set of moves that its parent had with some set amount of random deviation, this is the mutated baby
  </li>
  <li>
    This, then repeats until the dots navigate to the target    
  </li>
  <li>
    The randomness of the moves introduces a form of exploration which for the most part, gets punished but sometimes gets rewarded in the form of becoming the new "Parent" 
  </li>
</ul>
<em>Can double or half time with the up or down arrows respectively</em>
  
