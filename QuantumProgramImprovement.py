import quantum_ast  # module to be made later
import random
from typing import List
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


# Define the Problem
class ProgramImprovementProblem(FloatProblem):
    def __init__(self):
        super(ProgramImprovementProblem, self).__init__()
        self.target_value =   # Add a desired program output
        self.variable_names = ["a", "b", "n"]
        self.operators = [("ID"), ("CU"), ("SR")]

        self.number_of_variables = 10
        self.number_of_objectives = 1
        self.number_of_constraints = 1

        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = [len(self.operators) - 1] * self.number_of_variables

    def evaluate(self, solution: FloatSolution):
        program = self.decode(solution)
        accuracy = self.execute_program(program)
        solution.objectives[0] = accuracy # maximize

        # Check constraints
        if self.is_feasible(program):
            solution.attributes['feasible'] = True
        else:
            solution.attributes['feasible'] = False

    def decode(self, solution: FloatSolution) -> str:
        variables = solution.variables

        n = 0
        for i, bit in enumerate(variables):
            n += bit * (2 ** i)

        program = self.generate_program(n)

        return program

    def generate_program(n: int) -> str:
        if n == 0:
            return "ID (a, 0)"
        else:
            program = "rz_adder' (a, b) (N.of_nat {}) :=\n".format(n)
            program += "  match {} with\n".format(n)
            for i in range(1, n + 1):
                program += "  | 0 => ID (a, 0)\n".format(i)
                program += "  | S m => CU (a, {}) (SR m b); rz_adder' (a, b) m\n".format(i)
            program += "  end."
            return program

    def execute_program(self, program: str):
        # call Ocaml
        # need to come up with a test set
        return accuracy  # the number of correct output / total

    def is_feasible(self, program: str):
        try:
            # Try to parse the program using the quantum_ast module
            quantum_ast.parse(program)
            return True
        except SyntaxError:
            return False


# Define Mutation Operator
class ProgramMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float):
        super(ProgramMutation, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        random_index = random.randint(0, len(solution.variables) - 1)
        new_operator_index = random.randint(0, len(problem.operators) - 1)
        solution.variables[random_index] = new_operator_index
        return solution


# Configure and Run the Genetic Algorithm
if __name__ == "__main__":
    problem = ProgramImprovementProblem()
    algorithm = GeneticAlgorithm[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=100,
        max_evaluations=1000,
        mutation=ProgramMutation(probability=0.1),  # Can make other customized mutation
        crossover=Crossover(),  # Can make customized crossover
    )
    algorithm.run()

    solutions = algorithm.get_result()
    best_solution = min(solutions, key=lambda x: x.objectives[0])
    best_program = problem.decode(best_solution)
    best_output = problem.execute_program(best_program)

    print(f"Best Program: {best_program}")
    print(f"Best Output: {best_output}")
