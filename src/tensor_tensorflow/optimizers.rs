use crate::tensor_tensorflow::tensors_flow::FlowTensors;

/// Simple CPU implementations of common optimizers operating on FlowTensors (f32 only).
/// These are convenience implementations for examples and tests; they operate by
/// reading `FlowTensors::data()` and producing a new `FlowTensors` with updated values.

pub struct SGD { pub lr: f32 }
impl SGD {
    pub fn new(lr: f32) -> Self { SGD { lr } }
    pub fn step(&self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let p = params.data()?;
        let g = grads.data()?;
        if p.len() != g.len() { return None; }
        let mut out = Vec::with_capacity(p.len());
        for i in 0..p.len() { out.push(p[i] - self.lr * g[i]); }
        FlowTensors::new(&out, params.dims())
    }
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}
impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, m: Vec::new(), v: Vec::new(), t: 0 }
    }
    pub fn step(&mut self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let p = params.data()?;
        let g = grads.data()?;
        if p.len() != g.len() { return None; }
        if self.m.len() != p.len() { self.m = vec![0.0; p.len()]; self.v = vec![0.0; p.len()]; }
        self.t += 1;
        let mut out = Vec::with_capacity(p.len());
        let b1 = self.beta1 as f64; let b2 = self.beta2 as f64;
        let t = self.t as f64;
        for i in 0..p.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g[i] * g[i]);
            let m_hat = (self.m[i] as f64) / (1.0 - b1.powf(t));
            let v_hat = (self.v[i] as f64) / (1.0 - b2.powf(t));
            let step = self.lr as f64 * m_hat / (v_hat.sqrt() + self.eps as f64);
            out.push(p[i] - step as f32);
        }
        FlowTensors::new(&out, params.dims())
    }
}

pub struct Adagrad {
    pub lr: f32,
    eps: f32,
    sum_sq: Vec<f32>,
}
impl Adagrad {
    pub fn new(lr: f32) -> Self { Adagrad { lr, eps: 1e-8, sum_sq: Vec::new() } }
    pub fn step(&mut self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let p = params.data()?; let g = grads.data()?; if p.len() != g.len() { return None; }
        if self.sum_sq.len() != p.len() { self.sum_sq = vec![0.0; p.len()]; }
        let mut out = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            self.sum_sq[i] += g[i] * g[i];
            let adjusted = self.lr / ((self.sum_sq[i] + self.eps).sqrt());
            out.push(p[i] - adjusted * g[i]);
        }
        FlowTensors::new(&out, params.dims())
    }
}

pub struct RMSProp {
    pub lr: f32,
    pub rho: f32,
    pub eps: f32,
    mean_sq: Vec<f32>,
}
impl RMSProp {
    pub fn new(lr: f32) -> Self { RMSProp { lr, rho: 0.9, eps: 1e-8, mean_sq: Vec::new() } }
    pub fn step(&mut self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let p = params.data()?; let g = grads.data()?; if p.len() != g.len() { return None; }
        if self.mean_sq.len() != p.len() { self.mean_sq = vec![0.0; p.len()]; }
        let mut out = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            self.mean_sq[i] = self.rho * self.mean_sq[i] + (1.0 - self.rho) * g[i] * g[i];
            out.push(p[i] - self.lr * g[i] / (self.mean_sq[i].sqrt() + self.eps));
        }
        FlowTensors::new(&out, params.dims())
    }
}

pub struct Momentum {
    pub lr: f32,
    pub mu: f32,
    velocity: Vec<f32>,
}
impl Momentum {
    pub fn new(lr: f32, mu: f32) -> Self { Momentum { lr, mu, velocity: Vec::new() } }
    pub fn step(&mut self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let p = params.data()?; let g = grads.data()?; if p.len() != g.len() { return None; }
        if self.velocity.len() != p.len() { self.velocity = vec![0.0; p.len()]; }
        let mut out = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            self.velocity[i] = self.mu * self.velocity[i] + self.lr * g[i];
            out.push(p[i] - self.velocity[i]);
        }
        FlowTensors::new(&out, params.dims())
    }
}

pub struct Adadelta {
    pub rho: f32,
    pub eps: f32,
    eg: Vec<f32>,
    ed: Vec<f32>,
}
impl Adadelta {
    pub fn new() -> Self { Adadelta { rho: 0.95, eps: 1e-6, eg: Vec::new(), ed: Vec::new() } }
    pub fn step(&mut self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let p = params.data()?; let g = grads.data()?; if p.len() != g.len() { return None; }
        if self.eg.len() != p.len() { self.eg = vec![0.0; p.len()]; self.ed = vec![0.0; p.len()]; }
        let mut out = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            self.eg[i] = self.rho * self.eg[i] + (1.0 - self.rho) * g[i] * g[i];
            let update = - ( (self.ed[i] + self.eps).sqrt() / (self.eg[i] + self.eps).sqrt() ) * g[i];
            self.ed[i] = self.rho * self.ed[i] + (1.0 - self.rho) * update * update;
            out.push(p[i] + update);
        }
        FlowTensors::new(&out, params.dims())
    }
}

pub struct Ftrl {
    pub alpha: f32,
    pub beta: f32,
    pub l1: f32,
    pub l2: f32,
    z: Vec<f32>,
    n: Vec<f32>,
}
impl Ftrl {
    pub fn new(alpha: f32) -> Self { Ftrl { alpha, beta: 1.0, l1: 0.0, l2: 0.0, z: Vec::new(), n: Vec::new() } }
    pub fn step(&mut self, params: &FlowTensors, grads: &FlowTensors) -> Option<FlowTensors> {
        let _p = params.data()?; let g = grads.data()?; let len = g.len();
        if self.z.len() != len { self.z = vec![0.0; len]; self.n = vec![0.0; len]; }
        let mut w = vec![0.0_f32; len];
        for i in 0..len {
            let gi = g[i];
            let ni_old = self.n[i];
            self.n[i] += gi * gi;
            let sigma = (self.n[i].sqrt() - ni_old.sqrt()) / self.alpha;
            self.z[i] += gi - sigma * w[i];
            // compute proximal weight
            let zi = self.z[i];
            if zi.abs() <= self.l1 {
                w[i] = 0.0;
            } else {
                let sign = if zi < 0.0 { -1.0 } else { 1.0 };
                w[i] = (sign * (zi.abs() - self.l1)) / (self.l2 + (self.beta + self.n[i].sqrt()) / self.alpha);
            }
        }
        FlowTensors::new(&w, params.dims())
    }
}

// Re-export types at module root
// types are declared `pub` above; no extra re-exports needed here.
