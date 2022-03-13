import os
import pygame
import sys
import random
import math
import sys
sys.path.append("..")

from deep_hyperneat.genome import Genome
from deep_hyperneat.population import Population
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN
from deep_hyperneat.decode import decode

pygame.init()

# Windows
WIN_HEIGHT = 420
WIN_WIDTH = 960

FONT = pygame.font.Font(os.path.join("Assets/Others", "coolvetica rg.ttf"), 18)

SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "dino_running_01.png")),
           pygame.image.load(os.path.join("Assets/Dino", "dino_running_02.png"))]

SNEAKING = [pygame.image.load(os.path.join("Assets/Dino", "dino_sneaking_01.png")),
            pygame.image.load(os.path.join("Assets/Dino", "dino_sneaking_02.png"))]

JUMPING = pygame.image.load(os.path.join("Assets/Dino", "dino_jumping_01.png"))

BG = pygame.image.load(os.path.join("Assets/Others", "background.png"))

LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Obstacles", "large_cactus_01.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "large_cactus_02.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "large_cactus_03.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "large_cactus_04.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "large_cactus_05.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "large_cactus_06.png")),
                pygame.image.load(os.path.join("Assets/Obstacles", "large_cactus_07.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Obstacles", "small_cactus_01.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "small_cactus_02.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "small_cactus_03.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "small_cactus_04.png")),
                pygame.image.load(os.path.join(
                    "Assets/Obstacles", "small_cactus_05.png")),
                pygame.image.load(os.path.join("Assets/Obstacles", "small_cactus_06.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Obstacles", "bird_01.png")),
        pygame.image.load(os.path.join("Assets/Obstacles", "bird_02.png"))]


class Dino():
    X_POS = 100
    Y_POS = 307.5
    S_Y_POS = Y_POS + 17
    J_VEL = 30
    J_DISTANCE = 150
    J_HEIGHT = 65

    def __init__(self, image=RUNNING[0]):
        self.image = image
        self.alive = True
        self.dino_state = 0  # 0 = running | 1 = jumping | 2 = sneaking
        self.rect = pygame.Rect(self.X_POS, self.Y_POS,
                                image.get_width(), image.get_height())
        self.step_index = 0
        self.vy = 0
        self.isjumping = False
        self.gravity = 0
        self.speed = 0
        self.score = 0
        self.color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        self.eyes = [self.rect.x + 24, self.rect.y + 7]
        self.raytraces = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    def update(self, speed):
        self.speed = speed
        if(self.isjumping):
            self.vy += self.gravity
            self.rect.y += self.vy
            if(self.rect.y > self.Y_POS):
                self.isjumping = False
                self.rect.y = self.Y_POS
                self.vy = 0
            self.eyes = [self.rect.x + 28, self.rect.y + 5]
        else:
            if(self.dino_state == 0):
                self.run()
                self.eyes = [self.rect.x + 28, self.rect.y + 5]
            if(self.dino_state == 1):
                self.jump(speed)
                self.eyes = [self.rect.x + 28, self.rect.y + 5]
            if(self.dino_state == 2):
                self.sneak()
                self.eyes = [self.rect.x + 42, self.rect.y + 6]

        if(self.step_index > 10):
            self.step_index = 0
        else:
            self.step_index += 1

    def run(self):
        self.image = RUNNING[self.step_index // 6]
        self.rect = pygame.Rect(self.X_POS, self.Y_POS,
                                self.image.get_width(), self.image.get_height())
        self.rect.y = self.Y_POS

    def jump(self, speed):
        self.image = JUMPING
        if(self.isjumping == False):
            self.isjumping = True
            airtime = self.J_DISTANCE/speed
            self.gravity = (2*self.J_HEIGHT)/(airtime*airtime)
            self.vy = -self.gravity*airtime
            self.rect = pygame.Rect(
                self.X_POS, self.Y_POS, self.image.get_width(), self.image.get_height())
            self.rect.y += self.vy

    def sneak(self):
        self.image = SNEAKING[self.step_index // 6]
        self.rect = pygame.Rect(self.X_POS, self.Y_POS,
                                self.image.get_width(), self.image.get_height())
        self.rect.y = self.S_Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, self.color, (self.rect.x-1,
                         self.rect.y-1, self.rect.width+2, self.rect.height+2), 2)


class Obstacle:
    BIRD_HEIGHT = [290, 275, 275, 270, 250]
    LARGE_CACTUS_HEIGHT = 302.5
    SMALL_CACTUS_HEIGHT = LARGE_CACTUS_HEIGHT + 13

    def __init__(self, image, grounded):
        self.image = image
        self.grounded = grounded
        self.height = 0
        self.rect = self.image.get_rect()
        self.rect.x = WIN_WIDTH
        self.x = WIN_WIDTH
        self.step_index = 0

    def update(self, speed, obstacles):
        self.x -= speed
        if self.x < -self.rect.width:
            obstacles.pop()
        self.rect.x = self.x

        if(self.grounded == False):
            if(self.step_index > 10):
                self.step_index = 0
            else:
                self.step_index += 1

            index = self.step_index // 6
            if(index == 0):
                self.rect.y = self.height+6
            else:
                self.rect.y = self.height
            self.image = BIRD[index]

    def draw(self, SCREEN):
        SCREEN.blit(self.image, self.rect)


class LargeCactus(Obstacle):
    def __init__(self, image, grounded=True):
        super().__init__(image, grounded)
        self.height = self.LARGE_CACTUS_HEIGHT
        self.rect.y = self.height


class SmallCactus(Obstacle):
    def __init__(self, image, grounded=True):
        super().__init__(image, grounded)
        self.height = self.SMALL_CACTUS_HEIGHT
        self.rect.y = self.height


class Bird(Obstacle):
    def __init__(self, image, grounded=False):
        super().__init__(image, grounded)
        self.height = self.BIRD_HEIGHT[random.randint(0, 4)]
        self.rect.y = self.height


def distance(vector1, vector2):
    return int(math.sqrt((vector1[0]-vector2[0]) * (vector1[0]-vector2[0]) +
                         (vector1[1]-vector2[1]) * (vector1[1]-vector2[1])))

# Substrate parameters
sub_in_dims = [1,8]
sub_sh_dims = [1,3]
sub_o_dims = 2

# Evolutionary parameters
goal_fitness=10000
pop_key = 0
pop_size = 150
pop_elitism = 6 # ~ 5%
num_generations = 500

def evolution(genomes):
    clock = pygame.time.Clock()
    run = True
    inputs = [None] * 9
    outputs = [None] * 2
    obstacles = []
    client_data = []
    SPEED = 8
    SCORE = 0
    isbird = 0

    x_pos_bg = 0
    y_pos_bg = 332.5

    for genome in genomes:
        client_data.append([genome, Dino()])

    while run:
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                sys.exit()
        SCREEN.fill((255, 255, 255))
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (x_pos_bg+image_width, y_pos_bg))
        if(x_pos_bg <= -image_width):
            x_pos_bg = 0
        x_pos_bg -= SPEED
        latest_obstacle = 0
        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update(SPEED, obstacles)
            if(obstacle.rect.x > latest_obstacle):
                latest_obstacle = obstacle.rect.x
            for i in range(len(client_data)):
                if(client_data[i][1].rect.colliderect(obstacle.rect)):
                    if client_data[i][1].alive == True:
                        client_data[i][0][1].fitness = client_data[i][1].score
                        client_data[i][1].alive = False
        ALIVE = 0
        for i in range(len(client_data)):
            dinosaur = client_data[i][1]
            if dinosaur.alive == True:
                dinosaur.update(SPEED)
                dinosaur.draw(SCREEN)
                ALIVE += 1
        if(ALIVE == 0):
            break
        if((len(obstacles) == 0) or (latest_obstacle < WIN_WIDTH/2)):
            r_int = random.randint(0, 1028)
            if(r_int < 24):
                obstacles.insert(0, LargeCactus(
                    LARGE_CACTUS[random.randint(0, 6)]))
            elif(r_int < 40):
                obstacles.insert(0, SmallCactus(
                    SMALL_CACTUS[random.randint(0, 5)]))
            elif((r_int < 64) and (SPEED > 16)):
                obstacles.insert(0, Bird(BIRD[0]))
        SPEED += 0.0075
        SCORE += (0.3 + SPEED/8/3)
        for i in range(len(client_data)):
            dinosaur = client_data[i][1]
            target_obstacle = -1
            if(len(obstacles) > 0):
                if(obstacles[len(obstacles)-1].rect.x > dinosaur.rect.x):
                    target_obstacle = obstacles[len(obstacles)-1]
                elif(obstacles[0].rect.x > dinosaur.rect.x):
                    target_obstacle = obstacles[0]

            if dinosaur.alive == True:
                if target_obstacle != -1:
                    if not target_obstacle.grounded:
                        isbird = 1
                    else:
                        isbird = 0
                    dinosaur.score = SCORE
                    inputs = [distance(client_data[i][1].eyes, (target_obstacle.rect.x, target_obstacle.rect.y)),
                              SPEED, dinosaur.rect.y, target_obstacle.rect.x, target_obstacle.rect.y, 
                              target_obstacle.rect.w, target_obstacle.rect.h, isbird]
                    if isbird:
                        inputs.append(distance(((dinosaur.rect.x + dinosaur.rect.w), dinosaur.rect.y),
                                               (target_obstacle.rect.x, (target_obstacle.rect.y + target_obstacle.rect.h))))
                    else:
                        inputs.append(0)
                    cppn = CPPN.create(client_data[i][0][1])
                    substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
                    outputs = substrate.activate(inputs)
                    if (outputs[0] > 0.8):
                        dinosaur.dino_state = 1
                    elif (outputs[1] > 0.8):
                        dinosaur.dino_state = 2
                    else:
                        dinosaur.dino_state = 0
                else:
                    dinosaur.dino_state = 0

        msg_score = FONT.render(
            f'Score: {str(int(SCORE))}', True, (0, 0, 0))
        SCREEN.blit(msg_score, (30, 30))
        msg_speed = FONT.render(
            f'Speed: {str(int(SPEED*100/8) / 100)}', True, (0, 0, 0))
        SCREEN.blit(msg_speed, (30, 50))
        msg_generation = FONT.render(
            f'Generation: {pop.current_gen}', True, (0, 0, 0))
        SCREEN.blit(msg_generation, (30, 70))
        msg_alive = FONT.render(
            f'Alive: {str(ALIVE)}', True, (0, 0, 0))
        SCREEN.blit(msg_alive, (30, 90))
        if inputs[len(inputs) - 1] != None:
            for i in range(len(inputs)):
                msg = FONT.render(f'input[{i}]: {str(round(inputs[i], 2))}', True, (0, 0, 0))
                SCREEN.blit(msg, (WIN_WIDTH - 150, 18 * (i + 1)))
        if outputs[len(outputs) - 1] != None:
            for i in range(len(outputs)):
                msg = FONT.render(f'output[{i}]: {str(round(outputs[i], 2))}', True, (0, 0, 0))
                SCREEN.blit(msg, (WIN_WIDTH - 150, (len(inputs) * 18) + 36 + (18 * (i + 1))))
        clock.tick(30)
        pygame.display.update()


if __name__ == "__main__":
    # Inititalize population
    pop = Population(pop_key,pop_size,pop_elitism)

    # Run population on the defined task for the specified number of generations
    #	and collect the winner
    winner_genome = pop.run(evolution,goal_fitness,num_generations)

    # Run winning genome on the task again
    print("\nChampion Genome: {} with Fitness {}\n".format(winner_genome.key,
                                                    winner_genome.fitness))