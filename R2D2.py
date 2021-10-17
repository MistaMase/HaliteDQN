import hlt
import logging
import datetime
import math
import random

kDebug = False
kIsRushEnabled = True

kBotName = "R2D2"
kTimePerTurn = 1700 #per-turn time cutoff in milliseconds
kRadius = 20
kInfinity = 1000000000
kEpsilon = 0.0000001
kRushCutoff = 12
kNumTargetSearchIterations = 20
kRetreatCoeff = 0.6
kRetreatDistance = 5 * hlt.constants.MAX_SPEED
kNumRetreatIterations = 36
kDockedShipDangerRadius = 4 * hlt.constants.MAX_SPEED
kMyShipEnemyScanRadius = 4 * hlt.constants.MAX_SPEED
kCollisionAdjustmentEpsilon = 0.001
kEscapeFromEnemyShipDistance = 3 * hlt.constants.MAX_SPEED
#Is used in NavigateTarget to calculate closest_point
kMinDistToTargetDefault = 2
kMinDistToTargetRetreat = 0
#is used for choosing between central and closest planets in the beginning of the game
kClosestCentralPlanetDistanceCoeff = 3
#is used for choosing between largest and closest planets in the beginning of the game
kClosestLargestPlanetDistanceCoeff = 2.5
kPlayWithTheEnemyDist = 3.5 * hlt.constants.MAX_SPEED

kSafeToDockDistance = 1 * hlt.constants.MAX_SPEED
kShipSuicideHealthCutoff = 65
#It's actual value is six center-to-center, but we want to make sure all ships from our
#flock reach enemy ship with the attack, so we lower this value
kShipAttackRadius = 5
kEnemyShipsClusterRadius = 2 * hlt.constants.MAX_SPEED

#Is used to decide if it's profitable to rush
kRushIsPossibleDistance = 110
kRushIsPossibleDistance4p = 105

kConfirmRushDistance = 60


def PrintTurnTime(turn_start_time):
    if kDebug:
        current_time = datetime.datetime.utcnow()
        time_spent_in_milliseconds = (current_time - turn_start_time).total_seconds() * 1000
        logging.info("\ntime spent on turn {}: {} milliseconds".format(turn, time_spent_in_milliseconds))

def CheckTimeout(turn_start_time):    
    return (datetime.datetime.utcnow() - turn_start_time).total_seconds() * 1000 > kTimePerTurn        

def CollisionTime(entity1, target1, entity2, target2):
    velocity1 = [target1.x - entity1.x, target1.y - entity1.y]
    velocity2 = [target2.x - entity2.x, target2.y - entity2.y]
    p = [entity1.x - entity2.x, entity1.y - entity2.y]
    v = [velocity1[0] - velocity2[0], velocity1[1] - velocity2[1]]
    a = DotProduct(v, v)
    b = 2 * DotProduct(p, v)
    radius1 = hlt.constants.SHIP_RADIUS
    radius2 = entity2.radius if isinstance(entity2, hlt.entity.Planet) else hlt.constants.SHIP_RADIUS
    c = DotProduct(p, p) - pow((radius1 + radius2 + kCollisionAdjustmentEpsilon), 2)
    if abs(a) < kEpsilon:
        if abs(b) < kEpsilon:
            return c <= 0
        t = -c / b
        return (t >= 0) and (t <= 1)
    D = b * b - 4 * a * c
    if D < 0:
        return False
    sqrt_D = math.sqrt(D)
    t0 = (-b - sqrt_D) / (2 * a)
    t1 = (-b + sqrt_D) / (2 * a)
    t = min(t0, t1)
    return (t >= 0) and (t <= 1)


#Hardcode rush formation on turn 0
def RushFirstTurn(my_ships, enemy_ships):
    command_queue = list()
    #determine whether my ships are aligned vertically or horizontally
    vertically_aligned = abs(my_ships[0].x - my_ships[1].x) < kEpsilon
    if my_ships[0].x + 1 < enemy_ships[0].x:
        #i am on the left, opp is on the right
        if vertically_aligned:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
        else:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
    elif my_ships[0].x - 1 > enemy_ships[0].x:
        #i am on the right, opp is on the left
        if vertically_aligned:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
        else:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
    elif my_ships[0].y + 1 < enemy_ships[0].y:
        #my ships are at the top, opp is at the bottom
        if vertically_aligned:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 70))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 5, 88))
        else:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
    else:
        #i am at the bottom, opp is at the top
        if vertically_aligned:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
        else:
            command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
            command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
            command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))
    return command_queue    


def RushFirstTurn4p(game_map, my_ships, enemy_ships):
    my_ships.sort(key = lambda my_ship : my_ship.id)
    command_queue = list()
    #determine whether my ships are aligned vertically or horizontally
    #vertically_aligned = abs(my_ships[0].x - my_ships[1].x) < kEpsilon
    if my_ships[0].y < game_map.height / 2:
        #I am in the top part of the map
        command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 70))
        command_queue.append("t {} {} {}".format(my_ships[2].id, 0, 0))
        command_queue.append("t {} {} {}".format(my_ships[1].id, 5, 88))
    else:
        #I am in the bottom part of the map
        command_queue.append("t {} {} {}".format(my_ships[0].id, 3, 290))
        command_queue.append("t {} {} {}".format(my_ships[1].id, 0, 0))
        command_queue.append("t {} {} {}".format(my_ships[2].id, 5, 272))    
    return command_queue        

#Checks if given ship position is outside the game map boundaries
#Is used in EscapeFromEnemy to avoid getting stuck in the corners
def IsOutsideOfGameMap(ship, game_map):
    return ((ship.x < hlt.constants.SHIP_RADIUS + kEpsilon) 
        or (ship.x > game_map.width - hlt.constants.SHIP_RADIUS - kEpsilon)
        or (ship.y < hlt.constants.SHIP_RADIUS + kEpsilon)
        or (ship.y > game_map.height - hlt.constants.SHIP_RADIUS - kEpsilon))


def DistToClosestShip(ship, ships):
    min_dist = kInfinity
    for current_ship in ships:
        if ship.id != current_ship.id:
            min_dist = min(min_dist, ship.calculate_distance_between(current_ship))
    return min_dist

def DistToClosestEnemyShip(my_ship, enemy_ships):
    min_dist = kInfinity
    for enemy_ship in enemy_ships:
        min_dist = min(min_dist, my_ship.calculate_distance_between(enemy_ship))
    return min_dist


#returns True if ship collides with game map border, False otherwise
def CheckShipCollisionWithMapBorders(ship_position, game_map):
    offset = 3
    return  ((ship_position.x < offset + 0.5 + kEpsilon) or (ship_position.x > game_map.width - 0.5 - offset - kEpsilon) or 
        (ship_position.y < offset + 0.5 + kEpsilon) or (ship_position.y > game_map.height - 0.5 - offset - kEpsilon))


def ShipsInRadius(position, ships, radius):
    cnt = 0
    for ship in ships:
        if position.calculate_distance_between(ship) < radius:
            cnt += 1
    return cnt


def PlanetClosestToEnemyShips(planets, enemy_ships):
    closest_planet = None
    min_dist = kInfinity
    for planet in planets:
        dist = min([planet.calculate_distance_between(enemy_ship) for enemy_ship in enemy_ships])
        if dist < min_dist:
            min_dist = dist
            closest_planet = planet
    return (closest_planet, min_dist)


def PlanetsByDistance(game_map, ship):
    result = dict()
    for planet in game_map.all_planets():
        result.setdefault(ship.calculate_distance_between(planet), []).append(planet)
    return result

def ShipsByDistance(position, ships, ignore_docked_ships = True):
    result = dict()
    for ship in ships:
        if ignore_docked_ships and (ship.docking_status != ship.DockingStatus.UNDOCKED):
            result.setdefault(position.calculate_distance_between(ship), []).append(ship)
    return result    


def GetVector(entity1, entity2):
    return [entity2.x - entity1.x, entity2.y - entity1.y]

def LengthSquared(v):
    dx = v[0]
    dy = v[1]
    return dx * dx + dy * dy

def Length(v):
    return math.sqrt(LengthSquared(v))

def DotProduct(a, b):
    return a[0] * b[0] + a[1] * b[1]



def PlanetDangerIndex(planet, my_ships, enemy_ships):
    my_ship_cnt = 0
    enemy_ship_cnt = 0
    for my_ship in my_ships:
        if ((my_ship.docking_status == my_ship.DockingStatus.UNDOCKED) 
            and (planet.calculate_distance_between(my_ship) < kSafeToDockDistance)):
            my_ship_cnt += 1

    for enemy_ship in enemy_ships:
        if ((enemy_ship.docking_status == enemy_ship.DockingStatus.UNDOCKED) 
            and (planet.calculate_distance_between(enemy_ship) < kSafeToDockDistance)):
            enemy_ship_cnt += 1

    danger_index = enemy_ship_cnt - my_ship_cnt
    return danger_index    


def IsSafeToDock(planet, my_ships, enemy_ships):
    closest_enemy_ship = ShipClosestToPlanet(planet, enemy_ships)
    return closest_enemy_ship.calculate_distance_between(planet) > kSafeToDockDistance + planet.radius


def IsPlanetFull(planet, my_ships_to_planets_map):
    return len(planet._docked_ships) + my_ships_to_planets_map[planet.id] >= planet.num_docking_spots

def NearestPlanetCondition(entity, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
    return (isinstance(entity, hlt.entity.Planet) 
        and  (((not entity.is_owned()) or (entity.owner == game_map.get_me())) and not IsPlanetFull(entity, my_ships_to_planets_map))
        and IsSafeToDock(entity, my_ships, enemy_ships))


def ShipCondition(enemy_ship, game_map, enemy_ships_assigned_cnt):
    return (isinstance(enemy_ship, hlt.entity.Ship) 
        and (enemy_ship.owner != game_map.get_me()) 
        and CanAssignToEnemyShip(enemy_ship, enemy_ships_assigned_cnt))

def ClosestAvailablePlanet(ship, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
    ans = None
    min_dist = kInfinity
    for planet in game_map.all_planets():
        if NearestPlanetCondition(planet, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
            dist = ship.calculate_distance_between(planet)
            if dist < min_dist:
                min_dist = dist
                ans = planet
    return (ans, min_dist)

def ClosestDockedShip(ship, docked_ships):
    closest_docked_ship = None
    min_dist = kInfinity
    for docked_ship in docked_ships:
        dist = ship.calculate_distance_between(docked_ship)
        if dist < min_dist:
            min_dist = dist
            closest_docked_ship = docked_ship
    return closest_docked_ship


def ClosestFriendlyShip(ship, my_docked_ships, my_undocked_ships, radius = kInfinity, ignore_docked_ships = True):
    closest_friendly_ship = None
    min_dist = kInfinity
    for my_ship in my_ships:
        if ship.id == my_ship.id:
            continue
        if ignore_docked_ships and (my_ship.docking_status != my_ship.DockingStatus.UNDOCKED):
            continue
        dist = ship.calculate_distance_between(my_ship)
        if dist < min_dist:
            min_dist = dist
            closest_friendly_ship = my_ship
    return closest_friendly_ship

def ShipClosestToPlanet(planet, ships):
    closest_ship = None
    min_dist = kInfinity
    for ship in ships:
        dist = planet.calculate_distance_between(ship)
        if dist < min_dist:
            min_dist = dist
            closest_ship = ship
    return closest_ship

def ClosestEnemyShip(position, enemy_ships, ignore_docked_ships = True):
    closest_enemy_ship = None
    min_dist = kInfinity
    for enemy_ship in enemy_ships:
        if ignore_docked_ships and (enemy_ship.docking_status != enemy_ship.DockingStatus.UNDOCKED):
            continue
        if not CanAssignToEnemyShip(enemy_ship, enemy_ships_assigned_cnt):
            continue
        dist = position.calculate_distance_between(enemy_ship)
        if dist < min_dist:
            min_dist = dist
            closest_enemy_ship = enemy_ship
    return (closest_enemy_ship, min_dist)

def ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, ignore_docked_ships = True, ignore_undocked_ships = True):
    closest_enemy_ship = None
    min_dist = kInfinity
    for enemy_ship in enemy_ships:
        if ignore_docked_ships and (enemy_ship.docking_status != enemy_ship.DockingStatus.UNDOCKED):
            continue
        elif ignore_undocked_ships and (enemy_ship.docking_status == enemy_ship.DockingStatus.UNDOCKED):
            continue
        dist = min([enemy_ship.calculate_distance_between(my_ship) for my_ship in my_ships])
        if dist < min_dist:
            min_dist = dist
            closest_enemy_ship = enemy_ship
    return (closest_enemy_ship, min_dist)    

def MyShipClosestToEnemyShips(my_ships, enemy_ships):
    my_closest_ship = None
    min_dist = kInfinity
    for my_ship in my_ships:
        dist = min([my_ship.calculate_distance_between(enemy_ship) for enemy_ship in enemy_ships])
        if dist < min_dist:
            min_dist = dist
            my_closest_ship = my_ship
    return my_closest_ship


def MyShipClosestToEnemyShip(enemy_ship, my_ships):
    my_closest_ship = None
    min_dist = kInfinity
    for my_ship in my_ships:
        dist = my_ship.calculate_distance_between(enemy_ship)
        if dist < min_dist:
            min_dist = dist
            my_closest_ship = my_ship
    return my_closest_ship    

def AveragePosition(positions):
    x = 0
    y = 0
    cnt = 0
    for positions in positions:
        x += position.x
        y += position.y
        cnt += 1
    if cnt != 0:
        return hlt.entity.Position(x / cnt, y / cnt)
    return None

#calculates average position of ship's targets within given radius 
def AveragePositionOLD(ship, my_undocked_ships, my_ships_positions_next_turn, radius):
    x = 0
    y = 0
    cnt = 0
    for my_undocked_ship in my_undocked_ships:
        if ship.id == my_undocked_ship.id:
            continue
        target = my_ships_positions_next_turn[my_undocked_ship]
        if ship.calculate_distance_between(target) > radius:
            continue
        x += target.x
        y += target.y
        cnt += 1
    if cnt == 0:
        return hlt.entity.Position(ship.x, ship.y)
    return hlt.entity.Position(x / cnt, y / cnt)


def NumProducedShips(ships, produced_ships):
    for ship in ships:
        produced_ships.add(ship.id)
    return len(produced_ships)    


def Distractors(my_ships, my_docked_ships, my_undocked_ships,
    enemy_ships, enemy_docked_ships, enemy_undocked_ships):
    result = list()
    return result

def MyNavigate(ship, target, game_map, my_ships_positions_next_turn, enemy_ships,
                 speed, avoid_obstacles = True, max_corrections = 90, angular_step = 1, angular_sign = 1,
                 avoid_planets = True, avoid_friendly_ships = True, avoid_enemy_ships = False):
    if max_corrections <= 0:
        return None
    distance = ship.calculate_distance_between(target)
    angle = round(ship.calculate_angle_between(target))

    if avoid_obstacles:
        obstacle_dict = dict()
        if avoid_planets:
            for planet in game_map.all_planets():
                obstacle_dict[planet] = planet
        if avoid_friendly_ships:
            for my_ship, my_ship_position in my_ships_positions_next_turn.items():
                if ship.id != my_ship.id:
                    obstacle_dict[my_ship] = my_ship_position
        if avoid_enemy_ships:
            for enemy_ship in enemy_ships:
                obstacle_dict[enemy_ship] = enemy_ship

        for obstacle, obstacle_target in obstacle_dict.items():
            if CollisionTime(ship, target, obstacle, obstacle_target):
                new_angle = math.radians(round(angle + angular_step * angular_sign))
                new_target_dx = math.cos(new_angle) * distance
                new_target_dy = math.sin(new_angle) * distance
                new_target = hlt.entity.Position(ship.x + new_target_dx, ship.y + new_target_dy)
                return MyNavigate(ship, new_target, game_map, my_ships_positions_next_turn, enemy_ships,
                    speed, True, max_corrections - 1, angular_step + 1, -angular_sign)
        
        if CheckShipCollisionWithMapBorders(target, game_map):
                new_angle = math.radians(round(angle + angular_step * angular_sign))
                new_target_dx = math.cos(new_angle) * distance
                new_target_dy = math.sin(new_angle) * distance
                new_target = hlt.entity.Position(ship.x + new_target_dx, ship.y + new_target_dy)
                return MyNavigate(ship, new_target, game_map, my_ships_positions_next_turn, enemy_ships,
                    speed, True, max_corrections - 1, angular_step + 1, -angular_sign)
    return target


def EnemyHarrasingShips(my_docked_ships, enemy_undocked_ships):
    result = dict()
    for enemy_undocked_ship in enemy_undocked_ships:
        my_closest_docked_ship = None
        min_dist = 4 * hlt.constants.MAX_SPEED
        for my_docked_ship in my_docked_ships:            
            dist = enemy_undocked_ship.calculate_distance_between(my_docked_ship)
            if dist < min_dist:
                min_dist = dist
                my_closest_docked_ship = my_docked_ship
        if my_closest_docked_ship is not None:
            result[enemy_undocked_ship] = my_closest_docked_ship
    return result

def ProtectMyDockedShips(my_docked_ships, my_undocked_ships, my_ships_targets, enemy_harrasing_ships):
    for enemy_harrasing_ship, my_docked_ship in enemy_harrasing_ships.items():
        my_closest_undocked_ship = ClosestFriendlyShip(my_docked_ship, my_undocked_ships, my_docked_ships, ignore_docked_ships = True)
        if (my_closest_undocked_ship is not None) and (my_ships_targets[my_closest_undocked_ship] is None):
            min_approach_dist = 1
            my_ships_targets[my_closest_undocked_ship] = (my_docked_ship, min_approach_dist)


def HuntEnemyDockedShips(my_undocked_ships, my_ships_targets, enemy_docked_ships, enemy_undocked_ships):
    for enemy_docked_ship in enemy_docked_ships:
        cnt = 0
        for enemy_undocked_ship in enemy_undocked_ships:
            dist = enemy_undocked_ship.calculate_distance_between(enemy_docked_ship)
            if dist < 1 * hlt.constants.MAX_SPEED:
                cnt += 1
        if cnt <= 1:
            my_closest_undocked_ship = ClosestFriendlyShip(enemy_docked_ship, my_undocked_ships, my_docked_ships, ignore_docked_ships = True)
            my_ships_targets[my_closest_undocked_ship] = (enemy_docked_ship, 0)



def CanAssignToEnemyShip(enemy_ship, enemy_ships_assigned_cnt):
    cluster_size = ShipsInRadius(enemy_ship, enemy_ships, radius = kEnemyShipsClusterRadius)
    if cluster_size <= 3:
        return enemy_ships_assigned_cnt[enemy_ship.id] == 0
    # if cluster_size == 2:
    #     return enemy_ships_assigned_cnt[enemy_ship.id] <= 1
    return True

def FindEnemyShip(ship, my_ships, enemy_ships, enemy_ships_assigned_cnt):
    target_enemy_ship = None
    min_dist_to_docked_ship = kDockedShipDangerRadius
    for enemy_ship in enemy_ships:
        #if enemy ship is too far away ignore it
        if ship.calculate_distance_between(enemy_ship) > kMyShipEnemyScanRadius:
            continue
        if not CanAssignToEnemyShip(enemy_ship, enemy_ships_assigned_cnt):
            continue
        for my_docked_ship in my_ships:
            if my_docked_ship.docking_status != ship.DockingStatus.UNDOCKED:
                dist_to_docked_ship = my_docked_ship.calculate_distance_between(enemy_ship)
                if dist_to_docked_ship < min_dist_to_docked_ship:
                    min_dist_to_docked_ship = dist_to_docked_ship
                    target_enemy_ship = enemy_ship
        if target_enemy_ship is not None:
            return target_enemy_ship
        for my_ship in my_ships:
            dist_to_ship = my_ship.calculate_distance_between(enemy_ship)
            if dist_to_ship < min_dist_to_docked_ship:
                min_dist_to_docked_ship = dist_to_ship
                target_enemy_ship = enemy_ship
    return target_enemy_ship


#calculates the danger index for a docked ship by counting the number of active (undocked) enemy ships nearby
def DangerIndex(ship, my_ships, enemy_ships):
    my_ship_cnt = 0
    enemy_ship_cnt = 0
    for my_ship in my_ships:
        if ((my_ship.docking_status == my_ship.DockingStatus.UNDOCKED) 
            and (ship.calculate_distance_between(my_ship) < kSafeToDockDistance)):
            my_ship_cnt += 1

    for enemy_ship in enemy_ships:
        if ((enemy_ship.docking_status == enemy_ship.DockingStatus.UNDOCKED) 
            and (ship.calculate_distance_between(enemy_ship) < kSafeToDockDistance)):
            enemy_ship_cnt += 1

    danger_index = enemy_ship_cnt - my_ship_cnt
    return danger_index
 

def NavigateTarget(ship, target, game_map, my_ships_positions_next_turn, enemy_ships):
    closest_point = ship.closest_point_to(target[0], min_distance = target[1])
    dist_to_closest_point = int(ship.calculate_distance_between(closest_point))
    angle_to_closest_point = math.radians(round(ship.calculate_angle_between(closest_point)))

    dist_adjusted = min(hlt.constants.MAX_SPEED, int(dist_to_closest_point))

    closest_point_adjusted_x = ship.x + dist_adjusted * math.cos(angle_to_closest_point)
    closest_point_adjusted_y = ship.y + dist_adjusted * math.sin(angle_to_closest_point)
    closest_point_adjusted = hlt.entity.Position(closest_point_adjusted_x, closest_point_adjusted_y)
    dist_to_closest_point_adjusted = ship.calculate_distance_between(closest_point_adjusted)

    navigate_target = MyNavigate(
        ship,
        closest_point_adjusted,
        game_map,
        my_ships_positions_next_turn,
        enemy_ships,
        speed = min(int(hlt.constants.MAX_SPEED), round(dist_to_closest_point_adjusted)),
        avoid_obstacles = True, 
        max_corrections = 180, 
        angular_step = 1,
        angular_sign = 1,
        )
    return navigate_target



def ResolveCollisions(game_map, my_undocked_ships, enemy_ships, my_ships_positions_next_turn, 
    my_ships_targets, turn_start_time, num_iterations):

    resolved_collisions = {my_undocked_ship : False for my_undocked_ship in my_undocked_ships}
    for i in range(num_iterations):
        fail = False
        for ship in my_undocked_ships:
            if CheckTimeout(turn_start_time):
                return
            if resolved_collisions[ship]:
                continue
            target = my_ships_targets[ship]
            if target is not None:
                if isinstance(target[0], hlt.entity.Planet) and ship.can_dock(target[0]):
                    my_ships_positions_next_turn[ship] = hlt.entity.Position(ship.x, ship.y)
                else:
                    navigate_target = NavigateTarget(ship, target, game_map, my_ships_positions_next_turn, enemy_ships)
                    if navigate_target is None:
                        fail = True
                    else:
                        my_ships_positions_next_turn[ship] = navigate_target
                        resolved_collisions[ship] = True
        if not fail:
            break


def CheckRetreatMode(num_players, game_map, my_ships, enemy_ships):
    if num_players != 4:
        return False
    num_players_alive = 0
    for player in game_map.all_players():
        if len(player.all_ships()) > 0:
            num_players_alive += 1
    if num_players_alive < 3:
        return False

    (closest_enemy_ship, dist_to_closest_enemy_ship) = ClosestEnemyShip(my_ships[0], enemy_ships, ignore_docked_ships = False)
    if dist_to_closest_enemy_ship > 4 * hlt.constants.MAX_SPEED:
        return False

    if len(my_ships) == 1:
        if dist_to_closest_enemy_ship < 3 * hlt.constants.MAX_SPEED:
            return True

    if (len(my_ships) <= 2):
        enemy_ships_nearby = ShipsInRadius(my_ships[0], enemy_ships, 5 * hlt.constants.MAX_SPEED)
        if enemy_ships_nearby > 4:
            return True

    if (len(my_planets) > 0) and (len(my_planets) <= 2):
        enemy_ships_nearby = ShipsInRadius(my_planets[0], enemy_ships, my_planets[0].radius + 4 * hlt.constants.MAX_SPEED)
        if enemy_ships_nearby > 5:
            return True

    vacant_planets_cnt = len([planet for planet in game_map.all_planets() if (not planet.is_owned())])
    if vacant_planets_cnt > 6:
        return False

    return len(my_ships) < (len(enemy_ships) / num_players_alive * kRetreatCoeff)


def RetreatCondition(my_ship, enemy_ships):
    return True

def GameMapCorners(game_map):    
    top_left_corner = hlt.entity.Position(0, 0)
    top_right_corner = hlt.entity.Position(game_map.width, 0)
    bottom_left_corner = hlt.entity.Position(0, game_map.height)
    bottom_right_corner = hlt.entity.Position(game_map.width, game_map.height)
    return (top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner)

def RetreatTargetCorner(ship, game_map):
    top_left_corner = hlt.entity.Position(0, 0)
    top_right_corner = hlt.entity.Position(game_map.width, 0)
    bottom_left_corner = hlt.entity.Position(0, game_map.height)
    bottom_right_corner = hlt.entity.Position(game_map.width, game_map.height)
    corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]
    target_corner = None
    min_dist = kInfinity
    for corner in corners:
        dist = ship.calculate_distance_between(corner)
        if dist < min_dist:
            min_dist = dist
            target_corner = corner
    return target_corner



def EscapeFromEnemyShipsTarget(my_ship, game_map, enemy_ships):
    escape_position = None
    escape_angle = 0
    max_dist_to_closest_enemy = 0
    speed = hlt.constants.MAX_SPEED
    for angle in range(360):
        curr_angle_in_radians = math.radians(angle)
        curr_x = my_ship.x + math.cos(curr_angle_in_radians) * speed
        curr_y = my_ship.y + math.sin(curr_angle_in_radians) * speed
        curr_position = hlt.entity.Position(curr_x, curr_y)
        #Prevents getting stuck in the corner when running away from enemy ships
        if IsOutsideOfGameMap(curr_position, game_map):
            continue
        curr_dist_to_closest_enemy = DistToClosestEnemyShip(curr_position, enemy_ships)
        if curr_dist_to_closest_enemy > max_dist_to_closest_enemy:
            max_dist_to_closest_enemy = curr_dist_to_closest_enemy
            escape_angle = angle    
    escape_angle_in_radians = math.radians(escape_angle)
    escape_x = my_ship.x + math.cos(escape_angle_in_radians) * speed
    escape_y = my_ship.y + math.sin(escape_angle_in_radians) * speed
    return hlt.entity.Position(escape_x, escape_y)


def EscapeFromAllOtherShipsTarget(my_ship, game_map, my_ships, enemy_ships):
    escape_position = None
    escape_angle = 0
    max_dist_to_closest_ship = 0
    speed = hlt.constants.MAX_SPEED
    for angle in range(360):
        curr_angle_in_radians = math.radians(angle)
        curr_x = my_ship.x + math.cos(curr_angle_in_radians) * speed
        curr_y = my_ship.y + math.sin(curr_angle_in_radians) * speed
        curr_position = hlt.entity.Position(curr_x, curr_y)
        #Prevents getting stuck in the corner when running away from enemy ships
        if IsOutsideOfGameMap(curr_position, game_map):
            continue
        curr_dist_to_closest_ship = DistToClosestShip(curr_position, my_ships + enemy_ships)
        if curr_dist_to_closest_ship > max_dist_to_closest_ship:
            max_dist_to_closest_ship = curr_dist_to_closest_ship
            escape_angle = angle    
    escape_angle_in_radians = math.radians(escape_angle)
    escape_x = my_ship.x + math.cos(escape_angle_in_radians) * speed
    escape_y = my_ship.y + math.sin(escape_angle_in_radians) * speed
    return hlt.entity.Position(escape_x, escape_y)    


def ClosestCorner(my_ship, corners):
    closest_corner = corners[0]    
    min_dist = my_ship.calculate_distance_between(closest_corner)
    for i in range(1, 4):
        corner = corners[i]
        dist = my_ship.calculate_distance_between(corner)
        if dist < min_dist:
            min_dist = dist
            closest_corner = corner
    return (closest_corner, min_dist)




#this function returns the position that is as far away from enemy ships as possible
#it is used when we are losing the game and our ships are trying to survive
def RetreatTarget(my_ship, game_map, corners, enemy_ships):
    retreat_target = None
    dist_to_closest_enemy_ship = DistToClosestEnemyShip(my_ship, enemy_ships)
    if dist_to_closest_enemy_ship < kEscapeFromEnemyShipDistance:
        return EscapeFromEnemyShipsTarget(my_ship, game_map, enemy_ships)
    return RetreatTargetCorner(my_ship, game_map)
    

#returns True if it is profitable to try rush opponent 
def IsRushPossible(game_map, planets, my_ships, enemy_ships):
    if not kIsRushEnabled:
        return False

    if len(game_map.all_players()) == 4:
        (closest_enemy_ship, dist_to_closest_enemy_ship) = ClosestEnemyShip(my_ships[0], enemy_ships, ignore_docked_ships = True)
        return dist_to_closest_enemy_ship < kRushIsPossibleDistance4p

    initial_dist = my_ships[0].calculate_distance_between(enemy_ships[0])
    if initial_dist < kRushIsPossibleDistance:
        return True
    (planet_closest_to_enemy, enemy_dist) = PlanetClosestToEnemyShips(planets, enemy_ships)
    my_dist_to_planet = my_ships[0].calculate_distance_between(planet_closest_to_enemy)
    my_dist_to_enemy = my_ships[0].calculate_distance_between(enemy_ships[0])
    if my_dist_to_enemy < my_dist_to_planet:
        return False
    enemy_dist = enemy_ships[0].calculate_distance_between(planet_closest_to_enemy)
    my_dist = my_ships[0].calculate_distance_between(planet_closest_to_enemy)
    r = planet_closest_to_enemy.radius
    #Here plus one is for my turn 0, on which the rush formation is created
    my_time = (my_dist - r + 6) // 7 + 1
    enemy_time = (enemy_dist - r + 6) // 7
    is_rush_possible = (my_time  - enemy_time < 11)
    return is_rush_possible

def ConfirmRush2p(game_map, my_ships, enemy_ships, enemy_dist_to_center_turn_zero):
    game_map_center = hlt.entity.Position(game_map.width / 2, game_map.height / 2)
    return enemy_ships[0].calculate_distance_between(game_map_center) < enemy_dist_to_center_turn_zero

def MaxPairwiseDist(ships):
    max_pairwise_dist = 0
    for ship in ships:
        for another_ship in ships:
            if ship.id != another_ship.id:
                max_pairwise_dist = max(max_pairwise_dist, ship.calculate_distance_between(another_ship))
    return max_pairwise_dist


def ConfirmRushAgain2p(my_ships, my_docked_ships, my_undocked_ships,
    enemy_ships, enemy_docked_ships, enemy_undocked_ships):    
    if len(enemy_docked_ships) > 0:
        return True
    if (len(my_ships) >= len(enemy_ships)):
        if len(enemy_ships) == 1:
            return False
        return MaxPairwiseDist(enemy_ships) < 6 * hlt.constants.MAX_SPEED
    return True

def RushTargetPlayer4p(game_map, my_ships, enemy_ships):
    closest_enemy_ship = ClosestEnemyShip(my_ships[0], enemy_ships)[0]
    return closest_enemy_ship.owner

def EnemyContactRush4p(my_ships, enemy_ships):
    (closest_enemy_ship, dist_to_closest_enemy_ship) = ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, ignore_docked_ships = False, ignore_undocked_ships = False)
    return dist_to_closest_enemy_ship < 1 * hlt.constants.MAX_SPEED    

def ConfirmRush4p(my_ships, enemy_ships, rush_target_player_4p, rush_4p_enemy_contact, initial_rush_target_position_4p):
    if not rush_4p_enemy_contact:
        return True

    rush_target_ships = game_map._players[rush_target_player_4p.id].all_ships()
    rush_target_docked_ships = [ship for ship in rush_target_ships if (ship.docking_status != ship.DockingStatus.UNDOCKED)]
    if len(rush_target_docked_ships) > 0:
        return True
    if len(my_ships) > len(rush_target_ships):
        return len(rush_target_ships) > 1
    if MaxPairwiseDist(rush_target_ships) > 3 * hlt.constants.MAX_SPEED:
        return False
    (closest_enemy_ship, dist_to_closest_enemy_ship) = ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, ignore_docked_ships = False, ignore_undocked_ships = False)
    return dist_to_closest_enemy_ship < 5 * hlt.constants.MAX_SPEED


def PlanetsCloserToMeThanToEnemy(planets, my_ship, enemy_ship):
    return [planet for planet in planets if my_ship.calculate_distance_between(planet) < enemy_ship.calculate_distance_between(planet)]    

def CentralPlanets(game_map, planets):
    map_center = hlt.entity.Position(game_map.width / 2, game_map.height / 2)
    ans = sorted(planets, key = lambda planet: planet.calculate_distance_between(map_center))[0 : min(2, len(planets))]
    return ans


def ClosestCentralPlanet(my_ship, game_map, planets, my_ships, my_ships_to_planets_map, enemy_ships):
    closest_central_planet = None
    min_dist = kInfinity
    central_planets = CentralPlanets(game_map, planets)
    central_planets.sort(key = lambda planet : my_ships[0].calculate_distance_between(planet))
    closest_central_planet = central_planets[0]
    return (closest_central_planet, min_dist)

def LargePlanetClosestToCenter(my_ship, game_map, planets, my_ships, my_ships_to_planets_map, enemy_ships):
    game_map_center = hlt.entity.Position(game_map.width / 2, game_map.height / 2)
    closest_planet = None
    min_dist = kInfinity
    for planet in planets:
        if (planet.num_docking_spots >= 3) and NearestPlanetCondition(planet, game_map, my_ships, my_ships_to_planets_map, enemy_ships):            
            dist = game_map_center.calculate_distance_between(planet)
            if dist < min_dist:
                min_dist = dist
                closest_planet = planet
    return (closest_planet, min_dist)

def ClosestLargestPlanet(my_ship, game_map, planets, my_ships, my_ships_to_planets_map, enmey_ships):
    closest_largest_planet = None
    max_num_docking_spots = 0
    for planet in planets:
        if NearestPlanetCondition(planet, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
            num_docking_spots = planet.num_docking_spots
            if num_docking_spots > max_num_docking_spots:
                max_num_docking_spots = num_docking_spots
                closest_largest_planet = planet
    return (closest_largest_planet, min_dist)

def NotWorsePlanetClosestToMe(my_ship, reference_planet, planets):
    not_worse_planet = None
    min_dist = kInfinity
    for planet in planets:
        if planet.num_docking_spots >= reference_planet.num_docking_spots:
            dist = my_ship.calculate_distance_between(planet)
            if dist < min_dist:
                min_dist = dist
                not_worse_planet = planet
    return (not_worse_planet, min_dist)

def DangerousPosition(my_ship, target, my_ships,
    enemy_ships, enemy_docked_ships, enemy_undocked_ships):
    dist_to_target = my_ship.calculate_distance_between(target)
    adjusted_dist_to_target = min(hlt.constants.MAX_SPEED, round(dist_to_target))
    angle_to_target = my_ship.calculate_angle_between(target)
    angle_to_target_in_rad = math.radians(angle_to_target)
    predicted_x = my_ship.x + math.cos(angle_to_target_in_rad) * adjusted_dist_to_target
    predicted_y = my_ship.y + math.sin(angle_to_target_in_rad) * adjusted_dist_to_target
    my_ship_predicted_position = hlt.entity.Position(predicted_x, predicted_y)

    enemy_undocked_ships_near_predicted_position = ShipsInRadius(my_ship_predicted_position, enemy_undocked_ships, 2 * hlt.constants.MAX_SPEED)
    if enemy_undocked_ships_near_predicted_position == 0:
        return False
    if isinstance(target, hlt.entity.Ship):
        if DistToClosestShip(target, my_docked_ships) < 2 * hlt.constants.MAX_SPEED:
            return False
    dist_to_my_closest_ship = DistToClosestShip(my_ship, my_ships)
    if dist_to_my_closest_ship > hlt.constants.MAX_SPEED:
        return True
    return False


def InitialPlanetChoice2p(game_map, planets, my_ships, my_ships_to_planets_map, enemy_ships):    
    closest_planet, dist_to_closest_planet = ClosestAvailablePlanet(my_ships[0], game_map, my_ships, my_ships_to_planets_map, enemy_ships)
    target = closest_planet    
    if (closest_planet is not None) and (closest_planet.num_docking_spots >= 5):
        target = closest_planet
    else:    
        planets_closer_to_me = PlanetsCloserToMeThanToEnemy(planets, my_ships[0], enemy_ships[0])    
        (closest_central_planet, dist_to_closest_central_planet) = ClosestCentralPlanet(my_ships[0], game_map, planets_closer_to_me, my_ships, my_ships_to_planets_map, enemy_ships)
        if (closest_central_planet is not None) and (closest_central_planet.num_docking_spots >= 3):
            (not_worse_planet, dist_to_not_worse_planet) = NotWorsePlanetClosestToMe(my_ships[0], closest_central_planet, planets_closer_to_me)
            if (not_worse_planet is not None) and (not_worse_planet.num_docking_spots >= 4):
                target = not_worse_planet
            else:
                target = closest_central_planet
        else:            
            (large_planet_closest_to_center, dist_to_large_planet) = LargePlanetClosestToCenter(my_ships[0], game_map, planets_closer_to_me, my_ships, my_ships_to_planets_map, enemy_ships)
            if large_planet_closest_to_center is not None:
                (not_worse_planet, dist_to_not_worse_planet) = NotWorsePlanetClosestToMe(my_ships[0], large_planet_closest_to_center, planets_closer_to_me)
                if (not_worse_planet is not None) and (not_worse_planet.num_docking_spots >= 4):                
                    target = not_worse_planet
                elif (my_ships[0].calculate_distance_between(closest_central_planet) > not_worse_planet.calculate_distance_between(closest_central_planet)):
                    target = not_worse_planet
                else:
                    target = closest_central_planet
            else:
                target = closest_central_planet
    if NearestPlanetCondition(target, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
        return target
    return None


def InitialPlanetChoice4p(game_map, corners, planets, my_ships, my_ships_to_planets_map, enemy_ships):    
    planets_by_dist = [planet for planet in planets if (planet.num_docking_spots >= 3) and NearestPlanetCondition(planet, game_map, my_ships, my_ships_to_planets_map, enemy_ships)]
    if len(planets_by_dist) == 0:
        planets_by_dist = planets

    planets_by_dist.sort(key = lambda planet : my_ships[0].calculate_distance_between(planet))    
    if planets_by_dist[0].num_docking_spots >= 5:
        if planets_by_dist[0].calculate_distance_between(my_ships[0]) < 5 * hlt.constants.MAX_SPEED:
            return planets_by_dist[0]
    if len(planets_by_dist) > 1:
        if planets_by_dist[1].num_docking_spots >= 5:
            if planets_by_dist[1].calculate_distance_between(my_ships[0]) < 5 * hlt.constants.MAX_SPEED:
                return planets_by_dist[1]

    (closest_corner, dist_to_closest_corner) = ClosestCorner(my_ships[0], corners)
    planets_by_dist.sort(key = lambda planet : closest_corner.calculate_distance_between(planet))    
    game_map_center = hlt.entity.Position(game_map.width / 2, game_map.height / 2)

    if planets_by_dist[0].calculate_distance_between(game_map_center) < 5 * hlt.constants.MAX_SPEED:
        (closest_available_planet, dist_to_closest_available_planet) = ClosestAvailablePlanet(my_ships[0], game_map, my_ships, my_ships_to_planets_map, enemy_ships)
        if closest_available_planet is not None:
            return closest_available_planet
    if len(planets_by_dist) == 1:
        return planets_by_dist[0]
    return planets_by_dist[0] if (planets_by_dist[0].num_docking_spots + 1 >= planets_by_dist[1].num_docking_spots) else planets_by_dist[1]


def UsualStrategy(my_ship, turn, initial_planet_choice_2p, initial_planet_choice_4p, game_map, 
                my_ships, my_docked_ships, my_undocked_ships, planet_explorers,
                my_ships_to_planets_map, enemy_ships_assigned_cnt):

    target = FindEnemyShip(my_ship, my_ships, enemy_ships, enemy_ships_assigned_cnt)
    
    if (target is None) and (turn < 12):
        if (len(game_map._players) == 2):
            if NearestPlanetCondition(initial_planet_choice_2p, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
                target = initial_planet_choice_2p
        elif (len(game_map._players) == 4):
            if NearestPlanetCondition(initial_planet_choice_4p, game_map, my_ships, my_ships_to_planets_map, enemy_ships):
                target = initial_planet_choice_4p

    if target is None:        
        (closest_available_planet, dist_to_closest_available_planet) = ClosestAvailablePlanet(my_ship, game_map, my_ships, my_ships_to_planets_map, enemy_ships)
        (closest_enemy_ship, dist_to_closest_enemy_ship) = ClosestEnemyShip(my_ship, enemy_ships, ignore_docked_ships = False)
        if closest_available_planet is None:
            return closest_enemy_ship
        if closest_enemy_ship is None:
            return closest_available_planet
        
        if (dist_to_closest_available_planet < dist_to_closest_enemy_ship):
            target = closest_available_planet
        else:
            target = closest_enemy_ship

    if (turn <= 20) and (target is not None) and isinstance(target, hlt.entity.Planet):
        dist_to_planet = my_ship.calculate_distance_between(target)
        if dist_to_planet < hlt.constants.MAX_SPEED + target.radius:
            enemy_ships_nearby = ShipsInRadius(target, enemy_ships, target.radius + 6 * hlt.constants.MAX_SPEED)
            if enemy_ships_nearby > 0:
                (closest_enemy_ship, dist_to_closest_enemy_ship) = ClosestEnemyShip(my_ship, enemy_ships, ignore_docked_ships = False)
                target = closest_enemy_ship
    return target

def CheckEnemyContactRush(my_ships, enemy_ships):
    for my_ship in my_ships:
        min_dist_to_enemy_ship = min([my_ship.calculate_distance_between(enemy_ship) for enemy_ship in enemy_ships])
        if min_dist_to_enemy_ship < kShipAttackRadius:
            return True
    return False

def UpdateContactWithEnemyShips(my_ships, enemy_ships):
    for my_ship in my_ships:
        my_ship.contact_with_enemy_on_prev_turn = True


def RushEscapeTarget(game_map, my_ships, enemy_ships):
    return EscapeFromEnemyShipsTarget(my_ships[0], game_map, enemy_ships)


def RushStrategy(turn, game_map, num_players, planets, rush_target_player_4p,
                my_ships, my_docked_ships, my_undocked_ships,
                enemy_ships, enemy_docked_ships, enemy_undocked_ships,
                my_ships_targets, my_ships_positions_next_turn, my_ships_to_planets_map,
                turn_start_time, enemy_contact_rush_prev_turn):
    target = None
    enemy_contact = CheckEnemyContactRush(my_ships, enemy_ships)
    if enemy_contact:
        target = RushEscapeTarget(game_map, my_ships, enemy_ships)
    else:
        if num_players == 2:
            target = ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, False, True)[0]
            if target is None:
                target = ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, True, False)[0]
        else:
            target_enemy_ships = game_map._players[rush_target_player_4p.id].all_ships()
            target = ClosestEnemyShipToAllMyShips(my_ships, target_enemy_ships, False, True)[0]
            if target is None:
                target = ClosestEnemyShipToAllMyShips(my_ships, target_enemy_ships, True, False)[0]
                if target is None:
                    target = ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, False, True)[0]
                    if target is None:
                        target = ClosestEnemyShipToAllMyShips(my_ships, enemy_ships, True, False)[0]

    best_speed = 0
    best_angle = 0
    min_dist = kInfinity
    timeout_flag = False
    for curr_speed in range(hlt.constants.MAX_SPEED + 1):
        for curr_angle in range(360):
            if CheckTimeout(turn_start_time):
                timeout_flag = True
                break
            found_collision = False
            curr_angle_in_radians = math.radians(curr_angle)
            my_ships_positions = {my_ship.id : hlt.entity.Position(my_ship.x + curr_speed * math.cos(curr_angle_in_radians), my_ship.y + curr_speed * math.sin(curr_angle_in_radians)) for my_ship in my_ships}
            for my_ship in my_ships:
                for planet in planets:
                    if CollisionTime(my_ship, my_ships_positions[my_ship.id], planet, planet):
                        found_collision = True
                        break
                if (not found_collision) and CheckShipCollisionWithMapBorders(my_ships_positions[my_ship.id], game_map):
                    found_collision = True
                    break
            if not found_collision:                
                curr_dist = min([target.calculate_distance_between(my_ship_position) for my_ship_position in my_ships_positions.values()])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    best_speed = curr_speed
                    best_angle = curr_angle
        if timeout_flag:
            break
    min_dist_to_target = min([target.calculate_distance_between(my_ship) for my_ship in my_ships])
    if enemy_contact:
        pass
    elif (num_players == 2) and (len(my_ships) == 1) and (len(enemy_ships) == 1):
        pass
    elif target.docking_status == target.DockingStatus.UNDOCKED:
        if min_dist_to_target < hlt.constants.MAX_SPEED:
            best_speed = min(hlt.constants.MAX_SPEED, int(min_dist_to_target) - 3)
        elif min_dist_to_target < 2 * hlt.constants.MAX_SPEED:
            best_speed = min(hlt.constants.MAX_SPEED, best_speed)
    else:
        best_speed = min(hlt.constants.MAX_SPEED, int(min_dist_to_target) - 3)
    best_speed = max(best_speed, 0)
    command_queue = list()
    for my_ship in my_ships:
        command_queue.append("t {} {} {}".format(my_ship.id, best_speed, best_angle))
    return command_queue


def DetermineMyShipsTargets(turn, game_map, corners, num_players, my_ships, my_docked_ships, my_undocked_ships,
            my_ships_to_planets_map, my_ships_targets, my_docked_ships_by_planet, planet_explorers,
            num_my_produced_ships, initial_planet_choice_2p, initial_planet_choice_4p,
            enemy_ships, enemy_docked_ships, enemy_undocked_ships, enemy_harrasing_ships,
            num_enemy_produced_ships, enemy_ships_assigned_cnt,
            turn_start_time, is_rush_possible, is_retreat_mode_on):

    if is_retreat_mode_on:
        for my_ship in my_undocked_ships:
            if CheckTimeout(turn_start_time):
                break
            target = None
            (closest_corner, dist_to_closest_corner) = ClosestCorner(my_ship, corners)
            if dist_to_closest_corner < 2 * hlt.constants.MAX_SPEED:
                my_ships_near_corner = ShipsInRadius(closest_corner, my_ships, 3 * hlt.constants.MAX_SPEED)
                enemy_ships_near_corner = ShipsInRadius(closest_corner, enemy_ships, 5 * hlt.constants.MAX_SPEED)
                if (my_ships_near_corner > enemy_ships_near_corner) or ((enemy_ships_near_corner == 1) and (my_ships_near_corner == 1)):
                    target = ClosestEnemyShip(closest_corner, enemy_ships, ignore_docked_ships = True)[0]
                    approach_dist_to_target = 0
                else:
                    target = RetreatTarget(my_ship, game_map, corners, enemy_ships)
                    approach_dist_to_target = kMinDistToTargetRetreat
            else:
                target = RetreatTarget(my_ship, game_map, corners, enemy_ships)
                approach_dist_to_target = kMinDistToTargetRetreat
            if target is not None:
                my_ships_targets[my_ship] = [target, approach_dist_to_target]
        return



    ProtectMyDockedShips(my_docked_ships, my_undocked_ships, my_ships_targets, enemy_harrasing_ships)
    
    for my_ship in my_ships:
        if CheckTimeout(turn_start_time):
            return
        if my_ship.docking_status != my_ship.DockingStatus.UNDOCKED:
            continue
        if my_ships_targets[my_ship] is not None:
            continue

        target = None
        approach_dist_to_target = 0

        if my_ship.id in planet_explorers:
            target = planet_explorers[my_ship.id]
            approach_dist_to_target = kMinDistToTargetDefault

        
        if target is None:
            target = UsualStrategy(my_ship, turn, initial_planet_choice_2p, initial_planet_choice_4p, game_map, 
                my_ships, my_docked_ships, my_undocked_ships, planet_explorers,
                my_ships_to_planets_map, enemy_ships_assigned_cnt)
            approach_dist_to_target = kMinDistToTargetDefault
        if isinstance(target, hlt.entity.Ship) and (my_ship.health < kShipSuicideHealthCutoff):
            approach_dist_to_target = 0
                    
        if target is not None:
            my_ships_targets[my_ship] = [target, approach_dist_to_target]
            if isinstance(target, hlt.entity.Planet):
                my_ships_to_planets_map[target.id] += 1
            if isinstance(target, hlt.entity.Ship) and (target.owner != game_map.get_me()):
                enemy_ships_assigned_cnt[target.id] += 1



def DangerousPositionImproved(my_ship, game_map, my_ships, my_docked_ships, my_undocked_ships,
    my_ships_targets, my_ships_positions_next_turn,
    enemy_ships, enemy_docked_ships, enemy_undocked_ships):

    my_ship_position_next_turn = my_ships_positions_next_turn[my_ship]    
    enemy_undocked_ships_near_my_future_position = ShipsInRadius(my_ship_position_next_turn, enemy_undocked_ships, 2 * hlt.constants.MAX_SPEED)
    if enemy_undocked_ships_near_my_future_position == 0:
        return False
    target = my_ships_targets[my_ship][0]
    my_docked_ships_near_target = ShipsInRadius(target, my_docked_ships, hlt.constants.MAX_SPEED)
    if my_docked_ships_near_target > 0:
        return False
    my_undocked_ships_near_my_future_position = 0    
    for ship, position in my_ships_positions_next_turn.items():
        if position is None:
            continue
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue
        if ship.id == my_ship.id:
            continue
        if position.calculate_distance_between(my_ship_position_next_turn) < 3:
            my_undocked_ships_near_my_future_position += 1
    return my_undocked_ships_near_my_future_position < enemy_undocked_ships_near_my_future_position


def ClusterEscapeTarget(my_ship, my_ships_positions_next_turn):
    my_ship_closest_position = None
    my_closest_ship = None
    min_dist = kInfinity
    for ship, position in my_ships_positions_next_turn.items():
        if position is None:
            continue
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue
        if ship.id == my_ship.id:
            continue
        dist = position.calculate_distance_between(my_ship)
        if dist < min_dist:
            min_dist = dist
            my_closest_ship = ship
            my_ship_closest_position = position

    min_approach_dist = 1
    return [my_ship_closest_position, min_approach_dist]

def EscapeFromTargetPosition(my_ship, target):
    x = my_ship.x - target.x
    y = my_ship.y - target.y
    v = [x,y]
    len_v = Length(v)
    if abs(len_v) < kEpsilon:
        return my_ship
    v[0] /= len_v
    v[1] /= len_v
    v[0] *= hlt.constants.MAX_SPEED
    v[1] *= hlt.constants.MAX_SPEED
    return hlt.entity.Position(my_ship.x + v[0], my_ship.y + v[1])

def ClusterEscapeTargetImproved(my_ship, target, my_docked_ships, my_undocked_ships, my_ships_positions_next_turn):
    my_undocked_ships_nearby = [ship for ship in my_undocked_ships if my_ship.calculate_distance_between(ship) < 2 * hlt.constants.MAX_SPEED]
    escape_x = 0
    escape_y = 0
    cnt = 0
    for ship in my_undocked_ships_nearby:
        if ship.id != my_ship.id:
            escape_x += my_ships_positions_next_turn[ship].x
            escape_y += my_ships_positions_next_turn[ship].y
            cnt += 1
    if cnt == 0:
        return EscapeFromTargetPosition(my_ship, target)
    return hlt.entity.Position(escape_x / cnt, escape_y / cnt)




#Go over positions of my ships next turn and correct those which are "dangerous"
#Also should help clustering of my ships
def CheckMyShipsDangerousTargets(turn, game_map, num_players, 
    my_ships, my_docked_ships, my_undocked_ships,
    my_ships_to_planets_map, my_ships_targets, num_my_produced_ships,
    enemy_ships, enemy_docked_ships, enemy_undocked_ships, 
    num_enemy_produced_ships, enemy_ships_assigned_cnt,
    turn_start_time, is_rush_possible, is_retreat_mode_on):

    for my_ship, my_ship_position_next_turn in my_ships_positions_next_turn.items():
        if CheckTimeout(turn_start_time):
            return
        if my_ship.docking_status != my_ship.DockingStatus.UNDOCKED:
            continue
        target_tuple = my_ships_targets[my_ship]
        if target_tuple is None:
            continue        
        target = target_tuple[0]
        if DangerousPositionImproved(my_ship, game_map, my_ships, my_docked_ships, my_undocked_ships,
            my_ships_targets, my_ships_positions_next_turn,
            enemy_ships, enemy_docked_ships, enemy_undocked_ships):

            escape_target = ClusterEscapeTargetImproved(my_ship, target, my_docked_ships, my_undocked_ships, my_ships_positions_next_turn)
            approach_dist_to_target = 1
            if escape_target is not None:
                my_ships_targets[my_ship] = [escape_target, approach_dist_to_target]


def FillCommandQueue(my_ships_positions_next_turn, my_ships_targets):
    command_queue = list()
    for ship, position in my_ships_positions_next_turn.items():
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue
        target = None
        if my_ships_targets[ship] is not None:
            target = my_ships_targets[ship][0]
        if position is None:
            continue        
        if (abs(position.x - ship.x) < kEpsilon) and (abs(position.y - ship.y) < kEpsilon) and isinstance(target, hlt.entity.Planet):
            command_queue.append("d {} {}".format(ship.id, target.id))
        else:
            dist = ship.calculate_distance_between(position)
            speed = min(hlt.constants.MAX_SPEED, round(dist))
            angle = round(ship.calculate_angle_between(position))
            command_queue.append("t {} {} {}".format(ship.id, speed, angle))
    return command_queue            
    
def SendCommandQueue(game, command_queue):                    
    game.send_command_queue(command_queue)


#=========================================================================================================================
#=========================================================================================================================
#=========================================================================================================================
#=========================================================================================================================
#=========================================================================================================================


# GAME START

game = hlt.Game(kBotName)
logging.info("Starting my {} bot!".format(kBotName))


enemy_ships_positions_prev_turn = dict()
enemy_ships_positions_curr_turn = dict()

turn = 0    

num_players = 0

is_rush_possible = False
is_retreat_mode_on = False
#Number of ships I produced
my_produced_ships = set()
#Is used to counter rush in 1v1 games
enemy_produced_ships = set()

planet_explorers = dict()

enemy_contact_rush_prev_turn = False
initial_planet_choice_2p = None
initial_planet_choice_4p = None

enemy_dist_to_center_turn_zero = 0
rush_target_player_4p = None
rush_4p_enemy_contact = False
initial_rush_target_position_4p = None




while True:
    if kDebug:
        logging.info("==================================== turn {} =======================================".format(turn))
    
    turn_start_time = datetime.datetime.utcnow()

    game_map = game.update_map()
    players = game_map.all_players()    
    planets = game_map.all_planets()
    my_planets = [planet for planet in planets if (planet.owner == game_map.get_me())]
    enemy_ships = list()
    my_ships = list()
    my_protected_docked_ships = set()

    for player in players:
        player_ships = player.all_ships()
        if player == game_map.get_me():
            my_ships = player_ships
        else:
            enemy_ships += player_ships

    #Subtract three to account for initial ships that were given
    num_my_produced_ships = NumProducedShips(my_ships, my_produced_ships) - 3
    num_enemy_produced_ships = NumProducedShips(enemy_ships, enemy_produced_ships) - 3

    if kDebug:
        logging.info("\nMy ships at the start of the turn: {}".format(my_ships))

    my_docked_ships = [ship for ship in my_ships if ship.docking_status != ship.DockingStatus.UNDOCKED]
    enemy_docked_ships = [ship for ship in enemy_ships if ship.docking_status != ship.DockingStatus.UNDOCKED]
    my_undocked_ships = [ship for ship in my_ships if ship.docking_status == ship.DockingStatus.UNDOCKED]
    enemy_undocked_ships = [ship for ship in enemy_ships if ship.docking_status == ship.DockingStatus.UNDOCKED]
    my_ships_positions_next_turn = {ship : hlt.entity.Position(ship.x, ship.y) for ship in my_ships}
    my_ships_to_planets_map = {planet.id : 0 for planet in planets}
    my_ships_targets = {ship : None for ship in my_undocked_ships}            
    enemy_ships_assigned_cnt = {enemy_ship.id : 0 for enemy_ship in enemy_ships}

    my_docked_ships_by_planet = {planet : [ship for ship in my_docked_ships if ship.calculate_distance_between(planet) < planet.radius + hlt.constants.DOCK_RADIUS + kEpsilon] for planet in my_planets}


    enemy_harrasing_ships = EnemyHarrasingShips(my_docked_ships, enemy_undocked_ships)
    if kDebug:
        logging.info("\nmy_docked_ships: {}".format(my_docked_ships))
        logging.info("\nenemy_undocked_ships: {}".format(enemy_undocked_ships))
        logging.info("\nenemy_harrasing_ships: {}".format(enemy_harrasing_ships))
    





    if (turn == 0):
        if kIsRushEnabled:
            is_rush_possible = IsRushPossible(game_map, planets, my_ships, enemy_ships)
        num_players = len(game_map.all_players())
        corners = GameMapCorners(game_map)
        closer_to_me_planets = PlanetsCloserToMeThanToEnemy(planets, my_ships[0], enemy_ships[0])        
        if (num_players == 2):
            game_map_center = hlt.entity.Position(game_map.width / 2, game_map.height / 2)
            enemy_dist_to_center_turn_zero = enemy_ships[0].calculate_distance_between(game_map_center)
            initial_planet_choice_2p = InitialPlanetChoice2p(game_map, closer_to_me_planets, my_ships, my_ships_to_planets_map, enemy_ships)
        elif (num_players == 4):
            initial_planet_choice_4p = InitialPlanetChoice4p(game_map, corners, closer_to_me_planets, my_ships, my_ships_to_planets_map, enemy_ships)
            if kIsRushEnabled:               
                rush_target_player_4p = RushTargetPlayer4p(game_map, my_ships, enemy_ships)
                initial_rush_target_position_4p = rush_target_player_4p.all_ships()[0]
                if kDebug:
                    logging.info("\nrush_target_player_4p: {}".format(rush_target_player_4p))

    if not is_retreat_mode_on:
        is_retreat_mode_on = CheckRetreatMode(num_players, game_map, my_ships, enemy_ships)

    if is_rush_possible:
        if turn == 0:
            if num_players == 2:
                command_queue = RushFirstTurn(my_ships, enemy_ships)
            else:
                command_queue = RushFirstTurn4p(game_map, my_ships, enemy_ships)
        else:
            command_queue = RushStrategy(turn, game_map, num_players, planets, rush_target_player_4p,
                my_ships, my_docked_ships, my_undocked_ships,
                enemy_ships, enemy_docked_ships, enemy_undocked_ships,
                my_ships_targets, my_ships_positions_next_turn, my_ships_to_planets_map,
                turn_start_time, enemy_contact_rush_prev_turn)
            if num_players == 4:
                if not rush_4p_enemy_contact:
                    rush_4p_enemy_contact = EnemyContactRush4p(my_ships, enemy_ships)
            if turn == 0:
                if num_players == 2:
                    is_rush_possible = ConfirmRush2p(game_map, my_ships, enemy_ships, enemy_dist_to_center_turn_zero)
            elif turn >= 10:
                if num_players == 2:
                    is_rush_possible = ConfirmRushAgain2p(my_ships, my_docked_ships, my_undocked_ships,
                                            enemy_ships, enemy_docked_ships, enemy_undocked_ships)
                elif num_players == 4:
                    is_rush_possible = ConfirmRush4p(my_ships, enemy_ships, rush_target_player_4p, rush_4p_enemy_contact, initial_rush_target_position_4p)
    else:
        DetermineMyShipsTargets(turn, game_map, corners, num_players, my_ships, my_docked_ships, my_undocked_ships,
            my_ships_to_planets_map, my_ships_targets, my_docked_ships_by_planet, planet_explorers,
            num_my_produced_ships, initial_planet_choice_2p, initial_planet_choice_4p,
            enemy_ships, enemy_docked_ships, enemy_undocked_ships, enemy_harrasing_ships,
            num_enemy_produced_ships, enemy_ships_assigned_cnt,
            turn_start_time, is_rush_possible, is_retreat_mode_on)

        if kDebug:
            logging.info("\nmy_ships_targets: {}".format(my_ships_targets))        

        ResolveCollisions(game_map, my_undocked_ships, enemy_ships, my_ships_positions_next_turn, 
            my_ships_targets, turn_start_time, kNumTargetSearchIterations)

        if not is_retreat_mode_on:

            CheckMyShipsDangerousTargets(turn, game_map, num_players, 
                my_ships, my_docked_ships, my_undocked_ships,
                my_ships_to_planets_map, my_ships_targets, num_my_produced_ships,
                enemy_ships, enemy_docked_ships, enemy_undocked_ships, 
                num_enemy_produced_ships, enemy_ships_assigned_cnt,
                turn_start_time, is_rush_possible, is_retreat_mode_on)

            ResolveCollisions(game_map, my_undocked_ships, enemy_ships, my_ships_positions_next_turn, 
                my_ships_targets, turn_start_time, kNumTargetSearchIterations)

        if kDebug:
            logging.info("\nmy_ships_positions_next_turn: {}".format(my_ships_positions_next_turn))

        command_queue = FillCommandQueue(my_ships_positions_next_turn, my_ships_targets)

    enemy_contact_rush_prev_turn = CheckEnemyContactRush(my_ships, enemy_ships)
    
    SendCommandQueue(game, command_queue)
    PrintTurnTime(turn_start_time)
    turn += 1

    # TURN END
# GAME END