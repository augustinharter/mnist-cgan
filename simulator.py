import pygame
from random import random
import pymunk
import pymunk.pygame_util
from pygame.locals import *

class Simulator:
  def __init__(self):
    self.setup_pygame()

  def setup_pygame(self):
    pygame.init()
    self.size = 256
    self.screen = pygame.display.set_mode((self.size, self.size))
    pygame.display.set_caption("Phyre Task 00002:015")
    self.clock = pygame.time.Clock()
    self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

  def add_ball(self, radius, pos, color= (200, 0 , 0, 255)):
    space = self.space
    mass = radius / 5
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.2
    shape.color = color
    space.add(body, shape)
    return shape

  def setup_space(self):
    size = self.size
    space = pymunk.Space()
    space.gravity = (0.0, -1000.0)
    # Walls
    wall_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    wall_body.position = (0,0)
    wall_width = 1
    wd = pymunk.Segment(wall_body, (-1,      0), (size,      0), wall_width)
    wu = pymunk.Segment(wall_body, (-1, 1+size), (0, 1+size), wall_width)
    wl = pymunk.Segment(wall_body, (-1,      0), (-1,     size), wall_width)
    wr = pymunk.Segment(wall_body, (size,    0), (size,   size), wall_width)
    space.add(wall_body, wl, wr, wu, wd)

    # Segments
    segment_width = 5
    floor_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    wall_body.position = (0, 0)
    floor = pymunk.Segment(floor_body, (0, 3), (size, 3), segment_width)
    floor.color = (0, 0, 200, 255)
    floor.friction = 0.2

    plank_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    plank_body.position = (0, 156)
    plank = pymunk.Segment(plank_body, (0, 0), (3*size//5, 0), segment_width)
    plank.color = (0, 0, 0, 255)
    plank.friction = 0.2

    space.add(floor_body, floor, plank_body, plank)

    # Ball
    #ball = add_ball(space, 16, (size//3, 7.5*size//8), color = (0, 200, 0, 255))
    #add_ball(space, 16, (size//3 - 5, 5.5*size//8)) # One Solution
    self.space = space

  def add_rnd_ball(self):
    space = self.space
    action_pos = (random()*self.size, random()*self.size)
    action_radius = 5+random()*10
    while True:
      if not space.point_query_nearest(action_pos, action_radius+5, []):
        self.add_ball(action_radius, action_pos)
        break

  def run(self, n_frames=400, time_per_step=100):
    for step in range(n_frames):
      for event in pygame.event.get():
        if event.type == QUIT:
          sys.exit(0)
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
          sys.exit(0)
        
      self.space.step(time_per_step/7000)
      self.screen.fill((255,255,255))
      self.space.debug_draw(self.draw_options)
      pygame.display.flip()
      self.clock.tick(60)

  def quit(self):
    pygame.quit()


if __name__ == "__main__":
  sim = Simulator()
  sim.setup_space()
  sim.add_rnd_ball()
  sim.run()
  sim.quit()