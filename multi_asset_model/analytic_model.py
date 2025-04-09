import numpy as np
import torch

class analytic_solution():
    def find_values(self):
        # print(self.num_steps)
        dt = self.T/self.num_steps
        for i in range(1, self.num_steps + 1):
            # SigmaX = np.diag(self.sigma) @ self.corr @ np.diag(self.sigma).T
            SigmaX = self.SigmaOU @ self.SigmaOU.T

            dA = ((1/(2*self.ra)) * (np.diag(self.lambd)@self.mu).T @ np.linalg.inv(SigmaX) @ (np.diag(self.lambd)@self.mu))
            dA += (1/self.ra) * self.mu.T @ np.diag(self.lambd) @ self.B[i-1] 
            dA += (1-self.ra)/(2*self.ra) * self.B[i-1].T @ SigmaX @ self.B[i-1] + self.r
            dA += (1/2)* np.trace(SigmaX @ self.C[i-1])
            self.A[i] = self.A[i-1] + dt*dA

            dB = (((1-self.ra)/self.ra * SigmaX @ self.C[i-1] - np.diag(self.lambd)/self.ra - 
                   (1-self.ra)/self.ra *self.r * np.identity(self.n)).T @ self.B[i-1] -
                   (1/self.ra) * (np.diag(self.lambd) + self.r *np.identity(self.n)).T @np.linalg.inv(SigmaX)@(np.diag(self.lambd)@self.mu) +
                   (1/self.ra) * self.C[i-1].T @ np.diag(self.lambd) @ self.mu)
            self.B[i] = self.B[i-1] + dt*dB

            dC = ((1-self.ra)/self.ra * self.C[i-1].T @ SigmaX @ self.C[i-1] - 
                  (1/self.ra) * (np.diag(self.lambd)@self.C[i-1] + self.C[i-1].T @ np.diag(self.lambd)) - 
                  (2*self.r*(1-self.ra)/self.ra * self.C[i-1]) +
                  (1/self.ra)*(np.diag(self.lambd) + self.r*np.identity(self.n)).T @ 
                  np.linalg.inv(SigmaX) @ (np.diag(self.lambd)+self.r*np.identity(self.n)))
            self.C[i] = self.C[i-1] + dt*dC

    def __init__(self, n, ra, r, lambd, mu, SigmaOU, T, num_steps):
        self.n = n # number of assets
        self.ra = ra
        self.r = r
        self.lambd = np.array(lambd)
        self.mu = np.array(mu)
        self.SigmaOU = np.array(SigmaOU)
        self.T = T
        self.num_steps = num_steps
        self.A = {}
        self.B = {}
        self.C = {}

        self.A[0] = 0
        self.B[0] = np.zeros(n)
        self.C[0] = np.zeros((n,n))
        self.find_values()

    def no_cost_optimal(self, X, current_step):
        i = self.num_steps - current_step
        muW = (self.mu - X) * self.lambd / X
        SigmaW = np.diag(1/X) @ self.SigmaOU @ self.SigmaOU.T @ np.diag(1/X)
        SigmaWX = np.diag(1/X) @ self.SigmaOU @ self.SigmaOU.T
        
        pi = -(1/self.ra) * np.linalg.inv(SigmaW) @ ((self.r * np.ones(self.n) -muW) - (1-self.ra) * SigmaWX @
                (self.B[i] + self.C[i] @ X))
        return pi


    def no_cost_optimal_matrix_torch(self, X_matrix, current_step):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(X_matrix, np.ndarray):
            X_matrix = torch.tensor(X_matrix, dtype=torch.float32, device=device)
        else:
            X_matrix = X_matrix.to(device).float()

        # Helper function to convert NumPy arrays or scalars to float32 PyTorch tensors
        def to_tensor(x):
            if isinstance(x, np.ndarray) or isinstance(x, (int, float)):  # Handle scalars too
                return torch.tensor(x, dtype=torch.float32, device=device)
            return x.to(device).float()  # Ensure tensor is float32

        # Convert necessary parameters to float32 tensors
        self.SigmaOU = to_tensor(self.SigmaOU)
        self.mu = to_tensor(self.mu)
        self.lambd = to_tensor(self.lambd)
        self.ra = to_tensor(self.ra)  # Ensure ra is float32
        self.r = to_tensor(self.r)    # Ensure r is float32

        # Get current step index and extract B, C on the correct device and as float32
        i = self.num_steps - current_step
        B_i = to_tensor(self.B[i])  # Vector (shape: (n,))
        C_i = to_tensor(self.C[i])  # Matrix (shape: (n, n))

        m, n = X_matrix.shape  # Get batch size (m) and number of assets (n)

        # Compute muW
        muW = ((self.mu - X_matrix) * self.lambd / X_matrix).to(device)  # Shape: (m, n)
        # Compute SigmaW
        inv_X = torch.diag_embed(1 / X_matrix)  # Shape: (m, n, n)
        SigmaW = inv_X @ self.SigmaOU @ self.SigmaOU.T @ inv_X  # Shape: (m, n, n)

        # Compute SigmaWX
        SigmaWX = inv_X @ self.SigmaOU @ self.SigmaOU.T  # Shape: (m, n, n)

        # Compute inverse of SigmaW
        inv_SigmaW = torch.linalg.inv(SigmaW)  # Shape: (m, n, n)

        B_CX = B_i + torch.mm(C_i, X_matrix.T).T  # Shape: (m, n)

        term2 = torch.bmm(SigmaWX, B_CX.unsqueeze(2)).squeeze(2)  # Shape: (m, n)

        inner_term = (self.r * torch.ones((1, n), device=device)- muW) - (1 - self.ra) * term2  # Shape: (m, n)
        pi = - (1 / self.ra) * torch.bmm(inv_SigmaW, inner_term.unsqueeze(2)).squeeze(2)  # Shape: (m, n)

        return pi

    def J(self, W, X, num_steps):
        ra = self.ra
        return (W * (torch.exp(torch.tensor(self.A[num_steps], device='cuda', dtype=torch.float32) + torch.tensor(self.B[num_steps], device='cuda', dtype=torch.float32).T @ X + (X.T @ torch.tensor(self.C[num_steps], device='cuda', dtype=torch.float32) @ X / 2)))**(1 - ra) - 1)/(1 - ra)