import tkinter as tk
from tkinter import ttk, messagebox
import random
import heapq
import time
from collections import deque

ROWS = 15
COLS = 20
CELL = 36
PADDING = 12
DEFAULT_SPEED_MS = 30

COLOR_BG = "#f3f4f6"
COLOR_CELL = "#ffffff"
COLOR_WALL = "#2d3748"
COLOR_GRID = "#d1d5db"
COLOR_START = "#10b981"
COLOR_GOAL = "#ef4444"
COLOR_VISITED = "#93c5fd"
COLOR_PATH = "#fbbf24"
COLOR_BORDER = "#111827"

def neighbors(pos, rows, cols):
    r, c = pos
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)

def bfs(maze, start, goal, on_visit=None):
    rows, cols = len(maze), len(maze[0])
    q = deque([start])
    parent = {}
    visited = set([start])
    while q:
        cur = q.popleft()
        if on_visit:
            on_visit(cur)
        if cur == goal:
            break
        for nb in neighbors(cur, rows, cols):
            if maze[nb[0]][nb[1]] == 0 and nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                q.append(nb)
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return [], visited
    path.append(start)
    path.reverse()
    return path, visited

def dfs(maze, start, goal, on_visit=None):
    rows, cols = len(maze), len(maze[0])
    stack = [start]
    parent = {}
    visited = set([start])
    while stack:
        cur = stack.pop()
        if on_visit:
            on_visit(cur)
        if cur == goal:
            break
        for nb in neighbors(cur, rows, cols):
            if maze[nb[0]][nb[1]] == 0 and nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                stack.append(nb)
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return [], visited
    path.append(start)
    path.reverse()
    return path, visited

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(maze, start, goal, on_visit=None):
    rows, cols = len(maze), len(maze[0])
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    parent = {}
    gscore = {start: 0}
    visited = set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if on_visit:
            on_visit(cur)
        if cur == goal:
            break
        for nb in neighbors(cur, rows, cols):
            if maze[nb[0]][nb[1]] == 1:
                continue
            tentative = g + 1
            if tentative < gscore.get(nb, 1e9):
                gscore[nb] = tentative
                parent[nb] = cur
                heapq.heappush(open_heap, (tentative + heuristic(nb, goal), tentative, nb))
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return [], visited
    path.append(start)
    path.reverse()
    return path, visited

class MazeApp:
    def __init__(self, root):
        self.root = root
        root.title("Maze Visualizer — Pathfinding Algorithms")
        root.configure(bg=COLOR_BG)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TButton', padding=6, relief='flat', background="#111827", foreground="#fff")
        style.configure('TLabel', background=COLOR_BG, font=("Segoe UI", 10))
        style.configure('Header.TLabel', font=("Segoe UI", 12, 'bold'))

        self.left_frame = ttk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=(PADDING, 6), pady=PADDING)

        self.right_frame = ttk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=(6, PADDING), pady=PADDING, sticky='n')

        width = COLS * CELL + 2
        height = ROWS * CELL + 2
        self.canvas_container = tk.Canvas(self.left_frame, width=width + 2*4, height=height + 2*4,
        bg=COLOR_BORDER, highlightthickness=0)
        self.canvas_container.pack()
        self.canvas = tk.Canvas(self.canvas_container, width=width, height=height, bg=COLOR_CELL, highlightthickness=0)
        self.canvas.place(x=4, y=4)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.status = ttk.Label(root, text="Ready", style='TLabel')
        self.status.grid(row=1, column=0, columnspan=2, sticky='we', padx=PADDING, pady=(0, PADDING))

        ttk.Label(self.right_frame, text="Algorithm", style='Header.TLabel').pack(anchor='w')
        self.algo_var = tk.StringVar(value='A*')
        algo_menu = ttk.OptionMenu(self.right_frame, self.algo_var, 'A*', 'A*', 'BFS', 'DFS')
        algo_menu.pack(fill='x', pady=(2,8))

        ttk.Label(self.right_frame, text="Animation speed (ms)", style='Header.TLabel').pack(anchor='w', pady=(6,0))
        self.speed_var = tk.IntVar(value=DEFAULT_SPEED_MS)
        speed = ttk.Scale(self.right_frame, from_=5, to=300, orient='horizontal', command=self.on_speed_change, variable=self.speed_var)
        speed.pack(fill='x', pady=(2,8))

        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill='x', pady=(6,0))
        ttk.Button(btn_frame, text="Solve", command=self.on_solve).pack(fill='x', pady=4)
        ttk.Button(btn_frame, text="Reset Maze", command=self.reset_maze).pack(fill='x', pady=4)
        ttk.Button(btn_frame, text="Random Maze", command=self.random_maze).pack(fill='x', pady=4)

        self.draw_mode = tk.BooleanVar(value=False)
        draw_chk = ttk.Checkbutton(self.right_frame, text="Draw walls (click to toggle)", variable=self.draw_mode)
        draw_chk.pack(anchor='w', pady=(8,2))

        ttk.Label(self.right_frame, text="Legend", style='Header.TLabel').pack(anchor='w', pady=(10,2))
        legend = ttk.Frame(self.right_frame)
        legend.pack(anchor='w')
        self._legend_item(legend, COLOR_START, "Start")
        self._legend_item(legend, COLOR_GOAL, "Goal")
        self._legend_item(legend, COLOR_WALL, "Wall")
        self._legend_item(legend, COLOR_VISITED, "Visited")
        self._legend_item(legend, COLOR_PATH, "Path")

        self.rows = ROWS
        self.cols = COLS
        self.cell = CELL
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self._create_empty_maze()
        self.draw_grid()

    def _legend_item(self, parent, color, label):
        frm = ttk.Frame(parent)
        frm.pack(fill='x', pady=2)
        sw = tk.Canvas(frm, width=18, height=18, bg=COLOR_BG, highlightthickness=0)
        sw.create_rectangle(0,0,18,18, fill=color, outline='')
        sw.pack(side='left', padx=(0,6))
        ttk.Label(frm, text=label).pack(side='left')

    def _create_empty_maze(self):
        self.maze = [[0 for _ in range(self.cols)] for __ in range(self.rows)]
        for r in range(self.rows):
            self.maze[r][0] = 0
            self.maze[r][self.cols-1] = 0
        for c in range(self.cols):
            self.maze[0][c] = 0
            self.maze[self.rows-1][c] = 0

    def reset_maze(self):
        self._create_empty_maze()
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.draw_grid()
        self.set_status("Maze reset")

    def random_maze(self):
        self.maze = [[0 if random.random() > 0.25 else 1 for _ in range(self.cols)] for __ in range(self.rows)]
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.goal[0]][self.goal[1]] = 0
        self.draw_grid()
        self.set_status("Random maze generated")

    def set_status(self, txt):
        self.status.config(text=txt)

    def draw_grid(self, visited=None, path=None):
        self.canvas.delete("all")
        w = self.cell * self.cols
        h = self.cell * self.rows
        self.canvas.create_rectangle(0,0,w,h, fill=COLOR_CELL, outline='')
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = c * self.cell
                y0 = r * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                if self.maze[r][c] == 1:
                    fill = COLOR_WALL
                else:
                    fill = COLOR_CELL
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=COLOR_GRID)
        if visited:
            for (r, c) in visited:
                if (r,c) == self.start or (r,c) == self.goal:
                    continue
                x0 = c * self.cell
                y0 = r * self.cell
                self.canvas.create_rectangle(x0+4, y0+4, x0+self.cell-4, y0+self.cell-4, fill=COLOR_VISITED, outline='')

        if path:
            coords = []
            for (r, c) in path:
                cx = c * self.cell + self.cell/2
                cy = r * self.cell + self.cell/2
                coords.append((cx, cy))
                self.canvas.create_rectangle(c*self.cell+6, r*self.cell+6, c*self.cell+self.cell-6, r*self.cell+self.cell-6, fill=COLOR_PATH, outline='')
            if len(coords) > 1:
                flat = []
                for x,y in coords: flat.extend([x,y])
                self.canvas.create_line(*flat, width=6, fill="#b45309", capstyle='round', joinstyle='round')
        sr, sc = self.start
        gr, gc = self.goal
        self.canvas.create_rectangle(sc*self.cell+4, sr*self.cell+4, sc*self.cell+self.cell-4, sr*self.cell+self.cell-4, fill=COLOR_START, outline='')
        self.canvas.create_rectangle(gc*self.cell+4, gr*self.cell+4, gc*self.cell+self.cell-4, gr*self.cell+self.cell-4, fill=COLOR_GOAL, outline='')

    def on_canvas_click(self, event):
        c = int(event.x // self.cell)
        r = int(event.y // self.cell)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        if self.draw_mode.get():
            if (r,c) == self.start or (r,c) == self.goal:
                return
            self.maze[r][c] = 0 if self.maze[r][c] == 1 else 1
            self.draw_grid()
        else:
            mods = event.state
            if mods & 0x0001:
                self.start = (r, c)
                self.set_status(f"Start set to {self.start}")
                self.draw_grid()
            elif mods & 0x0004:
                self.goal = (r, c)
                self.set_status(f"Goal set to {self.goal}")
                self.draw_grid()
            else:
                if (r,c) == self.start or (r,c) == self.goal:
                    return
                self.maze[r][c] = 0 if self.maze[r][c] == 1 else 1
                self.draw_grid()

    def on_speed_change(self, _=None):
        pass

    def on_solve(self):
        algo = self.algo_var.get()
        speed = max(1, int(self.speed_var.get()))
        self.set_status(f"Solving with {algo} (speed={speed}ms)...")
        self.canvas.after(10, lambda: self._solve_and_animate(algo, speed))

    def _solve_and_animate(self, algo, speed_ms):
        visited_plot = set()
        def visit_callback(cell):
            visited_plot.add(cell)
            if len(visited_plot) % 6 == 0:
                self.draw_grid(visited=visited_plot)
                self.root.update()
                time.sleep(speed_ms / 1000.0)

        if algo == 'BFS':
            path, visited = bfs(self.maze, self.start, self.goal, on_visit=visit_callback)
        elif algo == 'DFS':
            path, visited = dfs(self.maze, self.start, self.goal, on_visit=visit_callback)
        else:
            path, visited = astar(self.maze, self.start, self.goal, on_visit=visit_callback)

        self.draw_grid(visited=visited)
        self.root.update()

        if path:
            for i in range(len(path)):
                sub = path[:i+1]
                self.draw_grid(visited=visited, path=sub)
                self.root.update()
                time.sleep(max(0.02, speed_ms/1000.0))
            self.set_status(f"Path found! length={len(path)}")
        else:
            self.set_status("No path found.")

def main():
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()