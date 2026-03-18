from collections import deque
import heapq

def bfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    queue = deque([start])
    visited = set([start])
    parent = {}

    while queue:
        r, c = queue.popleft()

        if (r, c) == goal:
            break

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr,nc))
                    parent[(nr,nc)] = (r, c)
                    queue.append((nr, nc))

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return []
    path.append(start)
    path.reverse()
    return path

def dfs(maze, start, goal):
    stack = [start]
    visited = set([start])
    parent = {}

    while stack:
        r, c = stack.pop()

        if (r, c) == goal:
            break

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr,nc)] = (r, c)
                    stack.append((nr, nc))

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return []
    path.append(start)
    path.reverse()
    return path

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    pq = [(0, start)]
    parent = {}
    g = {start: 0}

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            break

        r, c = current
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                new_cost = g[current] + 1
                if (nr, nc) not in g or new_cost < g[(nr,nc)]:
                    g[(nr,nc)] = new_cost
                    f = new_cost + heuristic((nr,nc), goal)
                    parent[(nr,nc)] = current
                    heapq.heappush(pq, (f, (nr,nc)))

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return []
    path.append(start)
    path.reverse()
    return path