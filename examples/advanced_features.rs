// Exemplo demonstrando as novas funcionalidades do ai_copper
use ai_copper::tensor_libortch::tensor::{Tensor, Linear, Optimizer};
use ai_copper::FlowTensors;
use ai_copper::tensor_tensorflow::tensors_flow::SparseTensor;
// Import otimizers implementados para o backend TensorFlow (FlowTensors)
use ai_copper::tensor_tensorflow::optimizers::{SGD, Adam, Adagrad, RMSProp, Momentum, Adadelta, Ftrl};

fn main() {
    // Demo principal: mostra várias funcionalidades oferecidas pela biblioteca
    // Comentários estão em português (pt-BR) explicando cada bloco.
    println!("=== AI Copper - Demo de Funcionalidades Avançadas ===\n");

    // ==================== CRIAÇÃO DE TENSORES ====================
    // Nesta seção criamos tensores básicos (zeros, randn, eye, ones, zeros_like)
    // Esses exemplos mostram as funções utilitárias para construir tensores.
    println!("1. CRIAÇÃO DE TENSORES");
    println!("{}", "-".repeat(50));
    
    let zeros = Tensor::zeros(2, 3);
    println!("Zeros (2x3):");
    zeros.print();
    
    let randn = Tensor::randn(2, 3);
    println!("\nRandn - Distribuição Normal (2x3):");
    randn.print();
    
    let eye = Tensor::eye(3);
    println!("\nEye - Matriz Identidade (3x3):");
    eye.print();
    
    let ones = Tensor::ones(2, 2);
    let zeros_like = ones.zeros_like();
    println!("\nZeros Like (mesma forma que ones):");
    zeros_like.print();
    
    // ==================== FUNÇÕES MATEMÁTICAS ====================
    // Aplicamos funções elementares (sin, exp, sqrt, pow) sobre tensores.
    // Cada operação retorna um novo tensor com o resultado elemento-a-elemento.
    println!("\n2. FUNÇÕES MATEMÁTICAS");
    println!("{}", "-".repeat(50));
    
    let x = Tensor::from_values(&[0.0, 1.0, 2.0, 3.0], 2, 2);
    println!("Tensor original:");
    x.print();
    
    let sin_x = x.sin();
    println!("\nSin(x):");
    sin_x.print();
    
    let exp_x = x.exp();
    println!("\nExp(x):");
    exp_x.print();
    
    let sqrt_abs = x.abs().sqrt();
    println!("\nSqrt(Abs(x)):");
    sqrt_abs.print();
    
    let pow_x = x.pow(2.0);
    println!("\nPow(x, 2):");
    pow_x.print();

    // Exemplos adicionais de LibTorch (operações úteis extras)
    println!("\nLibTorch extras:");
    // rand (uniform)
    let r = Tensor::rand(2, 3);
    println!("Rand (2x3):");
    r.print();

    // log - usar valores positivos para evitar -inf
    let pos = Tensor::from_values(&[0.1, 1.0, 2.0, 3.0], 2, 2);
    println!("\nLog(pos):");
    pos.log().print();

    // MatMul e Transpose: demonstra multiplicação matricial e transposição
    let a = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = Tensor::from_values(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    println!("\nMatMul a(2x3) x b(3x2):");
    let c = a.matmul(&b);
    c.print();

    println!("\nTranspose of a:");
    a.transpose().print();

    // NOTE: CrossEntropyLoss in libtorch expects a 0D or 1D target tensor
    // containing class indices (LongTensor). The small helper API here
    // creates 2D float matrices only, so creating a 2D target (shape [N,1])
    // causes a runtime error. To avoid the panic in this example we use
    // MSELoss with matching shapes instead. If you want to demo
    // CrossEntropyLoss later we can add a 1D/Long tensor creator helper.
    let logits = Tensor::from_values(&[2.0, 0.5, 0.1, 0.2, 1.5, 0.3], 3, 2); // 3 samples, 2 classes
    let targets = Tensor::from_values(&[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], 3, 2); // same shape as logits
    println!("\nMSELoss (used here instead of CrossEntropy to avoid dtype/shape mismatch):");
    let mse = logits.mse_loss(&targets);
    mse.print();

    // ==================== TENSORFLOW - FLOW TENSORS ====================
    // Demonstração das operações disponíveis no wrapper do TensorFlow
    // `FlowTensors` é a representação leve de tensores do backend TensorFlow.
    println!("\n6. TENSORFLOW - FLOW TENSORS (Exemplos de ops)");
    println!("{}", "-".repeat(50));

    // Criar um FlowTensor manualmente: passa dados (f32) e as dimensões (i64)
    // Usamos `expect` para deixar claro falhas em exemplos (mensagem legível).
    let ft = FlowTensors::new(&[0.0_f32, 1.0, 2.0, 3.0], &[2, 2])
        .expect("falha ao criar FlowTensors");
    // `data()` retorna Option<&[f32]> — aqui usamos `expect` para extrair o slice
    // e mostrar os valores. Em código de produção prefira tratar Option/Result.
    println!(
        "FlowTensor original (como slice): {:?}",
        ft.data().expect("falha ao obter dados do FlowTensors")
    );

    let ft_exp = ft.exp().expect("exp failed");
    println!(
        "Exp(ft): {:?}",
        ft_exp.data().expect("failed to get FlowTensors data")
    );

    let ft_ln = ft_exp.ln().expect("ln failed");
    println!(
        "Ln(Exp(ft)) (should approx equal original): {:?}",
        ft_ln.data().expect("failed to get FlowTensors data")
    );

    // log1p: calcula log(1 + x) elemento-a-elemento
    let ft_log1p = ft.log1p().expect("log1p falhou");
    println!("Log1p(ft): {:?}", ft_log1p.data().expect("falha ao obter dados"));

    // Sigmoid e Tanh: funções de ativação comuns
    let ft_sig = ft.sigmoid().expect("sigmoid falhou");
    println!("Sigmoid(ft): {:?}", ft_sig.data().expect("falha ao obter dados"));

    let ft_tanh = ft.tanh().expect("tanh falhou");
    println!("Tanh(ft): {:?}", ft_tanh.data().expect("falha ao obter dados"));

    // Novas ativações (FlowTensors)
    let ft_relu = ft.relu().expect("relu failed");
    println!("ReLU(ft): {:?}", ft_relu.data().expect("failed"));

    let ft_relu6 = ft.relu6().expect("relu6 failed");
    println!("ReLU6(ft): {:?}", ft_relu6.data().expect("failed"));

    let ft_elu = ft.elu().expect("elu failed");
    println!("ELU(ft): {:?}", ft_elu.data().expect("failed"));

    let ft_selu = ft.selu().expect("selu failed");
    println!("SELU(ft): {:?}", ft_selu.data().expect("failed"));

    // Softmax / LogSoftmax (1D example)
    let logits = FlowTensors::new(&[2.0_f32, 1.0, 0.1], &[3]).expect("failed to create logits");
    if let Some(sm) = logits.softmax(0) {
        println!("Softmax(logits): {:?}", sm.data().expect("failed"));
    }
    if let Some(lsm) = logits.log_softmax(0) {
        println!("LogSoftmax(logits): {:?}", lsm.data().expect("failed"));
    }

    // --- Novas operações: BiasAdd, BatchNormalization, SoftmaxCrossEntropy, Dropout ---
    println!("\nTensorFlow: BatchNormalization, BiasAdd, SoftmaxCrossEntropy, Dropout examples");
    println!("{}", "-".repeat(50));

    // BiasAdd: adicionar bias ao longo do eixo das colunas (axis=1)
    let x = FlowTensors::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("failed to create x");
    let bias = FlowTensors::new(&[0.1_f32, -0.2, 0.5], &[3]).expect("failed to create bias");
    if let Some(biased) = x.bias_add(&bias, 1) {
        println!("BiasAdd result dims={:?} data={:?}", biased.dims(), biased.data().expect("failed"));
    }

    // BatchNormalization (channels last example, axis=1 for [N,C])
    let bn_in = FlowTensors::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("failed bn_in");
    // note: pass data_format string to match the signature ("NHWC" for channels-last)
    if let Some(bn_out) = bn_in.batch_norm(None, None, None, None, 1e-5, 1, "NHWC") {
        println!("BatchNorm result dims={:?} data={:?}", bn_out.dims(), bn_out.data().expect("failed"));
    }

    // SoftmaxCrossEntropy: logits (3 samples, 2 classes) and one-hot labels
    let logits_ce = FlowTensors::new(&[2.0_f32, 0.5, 0.1, 0.2, 1.5, 0.3], &[3, 2]).expect("failed logits_ce");
    let labels_onehot = FlowTensors::new(&[0.0_f32, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2]).expect("failed labels");
    if let Some(loss) = FlowTensors::softmax_cross_entropy(&logits_ce, &labels_onehot, 1) {
        println!("SoftmaxCrossEntropy loss dims={:?} data={:?}", loss.dims(), loss.data().expect("failed"));
    }

    // Dropout: aplicar dropout com keep_prob=0.5 (treinamento)
    if let Some(dropped) = logits_ce.dropout(0.5, None) {
        println!("Dropout(keep=0.5) result dims={:?} data={:?}", dropped.dims(), dropped.data().expect("failed"));
    }

    // Conv2D example (NHWC): input 1x3x3x1, filter 2x2x1x1, VALID padding
    let inp = FlowTensors::new(&[
        1.0_f32, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0], &[1, 3, 3, 1]).expect("failed to create conv input");
    // Filter: 2x2, in_ch=1, out_ch=1 (simple sum over 2x2 regions)
    let filt = FlowTensors::new(&[1.0_f32, 1.0, 1.0, 1.0], &[2, 2, 1, 1]).expect("failed to create filter");
    if let Some(conv_res) = inp.conv2d(&filt, (1, 1), "VALID") {
        println!("Conv2D VALID result dims={:?} data={:?}", conv_res.dims(), conv_res.data().expect("failed"));
    }

    // MaxPool example
    if let Some(mp) = inp.max_pool((2, 2), (1, 1), "VALID") {
        println!("MaxPool VALID dims={:?} data={:?}", mp.dims(), mp.data().expect("failed"));
    }

    // AvgPool example
    if let Some(ap) = inp.avg_pool((2, 2), (1, 1), "VALID") {
        println!("AvgPool VALID dims={:?} data={:?}", ap.dims(), ap.data().expect("failed"));
    }

    // Funções trigonométricas (seno e cosseno)
    println!(
        "Sin(ft): {:?}",
        ft.sin()
            .expect("sin falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Cos(ft): {:?}",
        ft.cos()
            .expect("cos falhou")
            .data()
            .expect("falha ao obter dados")
    );

    // Comparações elemento-a-elemento
    // As funções retornam um tensor f32 contendo 1.0 para true e 0.0 para false
    let ft2 = FlowTensors::new(&[0.0_f32, 0.5, 2.0, 4.0], &[2, 2]).expect("falha ao criar ft2 FlowTensors");
    println!(
        "ft > ft2 : {:?}",
        ft.greater(&ft2)
            .expect("greater failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft < ft2 : {:?}",
        ft.less(&ft2)
            .expect("less failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft == ft2: {:?}",
        ft.equal(&ft2)
            .expect("equal failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft != ft2: {:?}",
        ft.not_equal(&ft2)
            .expect("not_equal failed")
            .data()
            .expect("failed to get FlowTensors data")
    );


    // Mais funções trigonométricas e inversas
    println!(
        "Tan(ft): {:?}",
        ft.tan()
            .expect("tan falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Asin(ft): {:?}",
        ft.asin()
            .expect("asin falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Acos(ft): {:?}",
        ft.acos()
            .expect("acos falhou")
            .data()
            .expect("falha ao obter dados")
    );
    println!(
        "Atan(ft): {:?}",
        ft.atan()
            .expect("atan falhou")
            .data()
            .expect("falha ao obter dados")
    );

    // Operações aritméticas entre FlowTensors
    let ft3 = FlowTensors::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("falha ao criar ft3 FlowTensors");
    println!(
        "ft + ft3: {:?}",
        ft.add(&ft3)
            .expect("add failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft - ft3: {:?}",
        ft.sub(&ft3)
            .expect("sub failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft * ft3: {:?}",
        ft.mul(&ft3)
            .expect("mul failed")
            .data()
            .expect("failed to get FlowTensors data")
    );
    println!(
        "ft / ft3: {:?}",
        ft.div(&ft3)
            .expect("div failed")
            .data()
            .expect("failed to get FlowTensors data")
    );

    // -------------------- NOVAS OPS TENSORFLOW --------------------
    println!("\nTENSORFLOW - NOVAS OPS: Fill, Pad/PadV2, Reverse, OneHot, Where");
    println!("{}", "-".repeat(50));

    // Fill: cria tensor 2x3 preenchido com 7.5
    let filled = FlowTensors::fill(&[2, 3], 7.5).expect("fill failed");
    println!("Fill 2x3 (7.5): {:?}", filled.data().expect("failed"));

    // Pad: adicionar 1 linha/coluna de zeros ao redor (paddings para cada eixo)
    let small = FlowTensors::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("failed to create small");
    // paddings = [(before_rows, after_rows), (before_cols, after_cols)]
    let padded = small.pad(&[(1, 1), (1, 1)]).expect("pad failed");
    println!("Small (2x2): {:?}", small.data().expect("failed"));
    println!("Padded (4x4 with zero pad): {:?}", padded.data().expect("failed"));

    // PadV2 com valor constante != 0
    let padded_const = small.pad_v2(&[(0, 1), (2, 0)], -1.0).expect("pad_v2 failed");
    println!("PadV2 (constant=-1) result: {:?}", padded_const.data().expect("failed"));

    // Reverse: inverter ao longo do eixo das colunas (axis=1)
    let rev_cols = small.reverse(&[1]).expect("reverse failed");
    println!("Reverse cols (axis=1): {:?}", rev_cols.data().expect("failed"));

    // OneHot: indices -> matriz one-hot
    let idxs = vec![0_i64, 2_i64, 1_i64];
    let oh = FlowTensors::one_hot(&idxs, 4, 1.0, 0.0).expect("one_hot failed");
    println!("OneHot (depth=4): {:?}", oh.data().expect("failed"));

    // Where: condicional elemento-a-elemento
    let cond = FlowTensors::new(&[1.0_f32, 0.0, 1.0, 0.0], &[2, 2]).expect("failed cond");
    let xa = FlowTensors::new(&[10.0_f32, 20.0, 30.0, 40.0], &[2, 2]).expect("failed xa");
    let yb = FlowTensors::new(&[0.0_f32, 1.0, 2.0, 3.0], &[2, 2]).expect("failed yb");
    let selected = FlowTensors::where_cond(&cond, &xa, &yb).expect("where failed");
    println!("Where(cond,x,y) result: {:?}", selected.data().expect("failed"));

    // -------------------- ARRAY OPS (Concat / Stack / Split / Slice / Gather / Transpose ND) --------------------
    println!("\nArray Ops: Concat, Stack/Unstack, Split, Slice, Gather, Transpose ND");
    println!("{}", "-".repeat(50));

    // Recreate two small tensors for array-op demos (avoid Clone on FlowTensors)
    let a_vals = small.data().expect("failed to get small data");
    let a = FlowTensors::new(a_vals, small.dims()).expect("failed to recreate a");
    let b_vals = filled.data().expect("failed to get filled data");
    // reshape filled to same shape as a for demo (if sizes differ, pick a compatible one)
    let b = FlowTensors::new(&b_vals[0..(a_vals.len())], a.dims()).expect("failed to recreate b");

    // Concat along axis 0 (stack rows)
    // Recreate separate FlowTensors instances from the same underlying data for safe ownership
    let a1 = FlowTensors::new(a.data().expect("failed"), a.dims()).expect("failed a1");
    let b1 = FlowTensors::new(b.data().expect("failed"), b.dims()).expect("failed b1");
    let mut v = Vec::new();
    v.push(a1);
    v.push(b1);
    if let Some(c) = FlowTensors::concat(&v, 0) {
        println!("Concat axis=0 dims={:?} data={:?}", c.dims(), c.data().expect("failed"));
    }

    // Concat along axis 1 (concatenate columns): tensors must have same number of rows
    // For 2D tensors shapes like [R, C1] and [R, C2] -> result [R, C1+C2]
    let a_col = FlowTensors::new(a.data().expect("failed"), a.dims()).expect("failed a_col");
    // build a compatible b with same rows but different columns by slicing/padding the filled tensor
    // here we reuse b which already has same shape as a in the demo; to demo axis=1, create a narrow tensor
    let b_cols = FlowTensors::new(&b.data().expect("failed")[0..(a.dims()[0] as usize)], &[a.dims()[0], 1]).expect("failed b_cols");
    if let Some(cat_axis1) = FlowTensors::concat(&[a_col, b_cols], 1) {
        println!("Concat axis=1 dims={:?} data={:?}", cat_axis1.dims(), cat_axis1.data().expect("failed"));
    } else {
        println!("Concat axis=1 failed: ensure tensors have same rank and matching dims except axis");
    }

    // Stack along a new leading axis
    let a1 = FlowTensors::new(a.data().expect("failed"), a.dims()).expect("failed a1");
    let b1 = FlowTensors::new(b.data().expect("failed"), b.dims()).expect("failed b1");
    if let Some(stacked) = FlowTensors::stack(&[a1, b1], 0) {
        println!("Stacked dims={:?} data={:?}", stacked.dims(), stacked.data().expect("failed"));
        // Unstack back
        if let Some(parts) = stacked.unstack(0) {
            for (i, p) in parts.iter().enumerate() {
                println!("Unstack part {} dims={:?} data={:?}", i, p.dims(), p.data().expect("failed"));
            }
        }
    }

    // Split along axis 0 into 2 equal parts (if compatible)
    if let Some(splits) = a.split(2, 0) {
        for (i, s) in splits.iter().enumerate() {
            println!("Split part {} dims={:?} data={:?}", i, s.dims(), s.data().expect("failed"));
        }
    }

    // Slice: take the first row of `a`
    if let Some(sliced) = a.slice(&[0, 0], &[1, a.dims()[1]]) {
        println!("Slice first row dims={:?} data={:?}", sliced.dims(), sliced.data().expect("failed"));
    }

    // Gather: collect rows 0 and 1 along axis 0
    if a.dims()[0] >= 2 {
        if let Some(gathered) = FlowTensors::gather(&a, &[0, 1], 0) {
            println!("Gather rows [0,1] dims={:?} data={:?}", gathered.dims(), gathered.data().expect("failed"));
        }
    }

    // Transpose ND: swap axes for 2D matrix
    if let Some(tnd) = a.transpose_nd(&[1, 0]) {
        println!("Transpose ND dims={:?} data={:?}", tnd.dims(), tnd.data().expect("failed"));
    }


    // Exemplo grande para acionar paralelismo interno (CPU-only)
    // (útil para demonstrar o uso de rayon em operações element-wise)
    let large_vals = vec![0.1_f32; 5000];
    let large_ft = FlowTensors::new(&large_vals, &[5000]).expect("falha ao criar large_ft FlowTensors");
    let large_exp = large_ft.exp().expect("falha ao executar exp() em large_ft");
    println!(
        "Large exp first element: {:?}",
        large_exp.data().expect("failed to get FlowTensors data")[0]
    );
    
    // ==================== SPARSE TENSORS (Sparse Ops) ====================
    println!("\n9. TENSORFLOW - SPARSE TENSORS (Simulated)");
    println!("{}", "-".repeat(50));

    // Construir dois SparseTensors 2x3 com alguns elementos não nulos
    // Indices are coordinates per element, values are f32, shape is [2,3]
    let indices_a = vec![vec![0_i64, 1_i64], vec![1, 2]];
    let values_a = vec![10.0_f32, 3.0_f32];
    let shape = vec![2_i64, 3_i64];
    let sp_a = SparseTensor::new(indices_a, values_a, shape.clone())
        .expect("failed to create sparse A");

    let indices_b = vec![vec![0_i64, 0_i64], vec![1, 2_i64]];
    let values_b = vec![1.5_f32, 2.5_f32];
    let sp_b = SparseTensor::new(indices_b, values_b, shape.clone())
        .expect("failed to create sparse B");

    // SparseAdd (materializes dense result)
    let added = sp_a.sparse_add(&sp_b).expect("sparse_add failed");
    println!("SparseAdd result (dense): {:?}", added.data().expect("failed to get data"));

    // SparseTensorDenseMatMul - create a dense matrix (KxN) to multiply
    // For this example treat sparse as shape [2,3], so K=3. Dense must be [3,2]
    let dense = FlowTensors::new(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2])
        .expect("failed to create dense mat");
    let matmul_res = sp_a.sparse_tensor_dense_matmul(&dense).expect("sparse matmul failed");
    println!("SparseTensorDenseMatMul result: {:?}", matmul_res.data().expect("failed to get data"));

    // SparseConcat along axis 0 (concatenate two sparse tensors into 4x3)
    let concat_res = SparseTensor::sparse_concat_axis0(&[sp_a.clone(), sp_b.clone()])
        .expect("sparse_concat failed");
    println!("SparseConcat (axis0) dense result: {:?}", concat_res.data().expect("failed to get data"));

    // SparseSlice: take first row of the concatenated tensor
    let slice_begin = &[0_i64, 0_i64];
    let slice_size = &[1_i64, 3_i64];
    // build SparseTensor from concat result by recreating sparse from dense (cheap way for example)
    let concat_dense = concat_res.data().expect("failed");
    // convert dense back to a SparseTensor representation
    let mut idxs = Vec::new();
    let mut vals = Vec::new();
    for i in 0..(concat_res.dims()[0] as usize) {
        for j in 0..(concat_res.dims()[1] as usize) {
            let v = concat_dense[i * (concat_res.dims()[1] as usize) + j];
            if v != 0.0 {
                idxs.push(vec![i as i64, j as i64]);
                vals.push(v);
            }
        }
    }
    let concat_sparse = SparseTensor::new(idxs, vals, concat_res.dims().to_vec())
        .expect("failed to build sparse from dense");
    let sliced = concat_sparse.sparse_slice(slice_begin, slice_size).expect("sparse_slice failed");
    println!("SparseSlice dense result: {:?}", sliced.data().expect("failed"));

    // SparseReshape: reshape slice to shape [3] (flatten) and show result
    let reshaped = SparseTensor::new(
        vec![vec![0_i64, 0_i64]], vec![1.0_f32], vec![1_i64, 3_i64]
    ).and_then(|s| s.sparse_reshape(&[3_i64]));
    if let Some(r) = reshaped {
        println!("SparseReshape result: {:?}", r.data().expect("failed"));
    }
    
    // ==================== OTIMIZADORES (Training Ops) - TensorFlow FlowTensors ====================
    println!("\n8. TENSORFLOW - OTIMIZADORES (Training Ops)");
    println!("{}", "-".repeat(50));

    // Exemplo simples: parâmetros e gradientes 1D
    let params = FlowTensors::new(&[0.5_f32, -0.3, 1.0], &[3]).expect("falha ao criar params");
    let grads = FlowTensors::new(&[0.1_f32, -0.2, 0.05], &[3]).expect("falha ao criar grads");

    println!("Params iniciais: {:?}", params.data().expect("failed"));
    println!("Grads: {:?}", grads.data().expect("failed"));

    // SGD
    let sgd = SGD::new(0.1);
    let params_sgd = sgd.step(&params, &grads).expect("sgd step failed");
    println!("After SGD step: {:?}", params_sgd.data().expect("failed"));

    // Adam (estadoful)
    let mut adam = Adam::new(0.1);
    let mut p = FlowTensors::new(params.data().expect("failed"), params.dims()).expect("failed to clone params");
    for i in 0..3 {
        p = adam.step(&p, &grads).expect("adam step failed");
        println!("After Adam step {}: {:?}", i + 1, p.data().expect("failed"));
    }

    // Adagrad
    let mut adg = Adagrad::new(0.1);
    let params_adg = adg.step(&params, &grads).expect("adagrad failed");
    println!("After Adagrad step: {:?}", params_adg.data().expect("failed"));

    // RMSProp
    let mut rms = RMSProp::new(0.01);
    let params_rms = rms.step(&params, &grads).expect("rmsprop failed");
    println!("After RMSProp step: {:?}", params_rms.data().expect("failed"));

    // Momentum
    let mut mom = Momentum::new(0.01, 0.9);
    let params_mom = mom.step(&params, &grads).expect("momentum failed");
    println!("After Momentum step: {:?}", params_mom.data().expect("failed"));

    // Adadelta
    let mut ada = Adadelta::new();
    let params_ada = ada.step(&params, &grads).expect("adadelta failed");
    println!("After Adadelta step: {:?}", params_ada.data().expect("failed"));

    // Ftrl (FTRL-Proximal implementado)
    let mut ftrl = Ftrl::new(0.1);
    let params_ftrl = ftrl.step(&params, &grads).expect("ftrl failed");
    println!("After Ftrl step: {:?}", params_ftrl.data().expect("failed"));

    // (In-place update demos removed per user request)

    
    // ==================== RANDOM OPS (TensorFlow FlowTensors) ====================
    println!("\n7. TENSORFLOW - RANDOM OPS (FlowTensors)");
    println!("{}", "-".repeat(50));

    // RandomUniform
    let ru = FlowTensors::random_uniform(&[2, 3], 0.0, 1.0).expect("random_uniform failed");
    println!("RandomUniform (2x3): {:?}", ru.data().expect("falha ao obter dados de RandomUniform"));

    // RandomNormal
    let rn = FlowTensors::random_normal(&[2, 3], 0.0, 1.0).expect("random_normal failed");
    println!("RandomNormal (2x3): {:?}", rn.data().expect("falha ao obter dados de RandomNormal"));

    // TruncatedNormal
    let tn = FlowTensors::truncated_normal(&[2, 3], 0.0, 1.0, 2.0).expect("truncated_normal failed");
    println!("TruncatedNormal (2x3): {:?}", tn.data().expect("falha ao obter dados de TruncatedNormal"));

    // RandomGamma
    let rg = FlowTensors::random_gamma(&[2, 3], 2.0, 1.0).expect("random_gamma failed");
    println!("RandomGamma (shape=2, scale=1) (2x3): {:?}", rg.data().expect("falha ao obter dados de RandomGamma"));

    // RandomShuffle (1D)
    let v = FlowTensors::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], &[5]).expect("falha ao criar FlowTensors v");
    let shuffled = v.random_shuffle().expect("falha ao embaralhar FlowTensors");
    println!("Original 1D: {:?}", v.data().expect("falha ao obter dados de v"));
    println!("Shuffled 1D: {:?}", shuffled.data().expect("falha ao obter dados de shuffled"));

    // Multinomial (amostrar 4 índices a partir de probabilidades)
    let probs = FlowTensors::new(&[0.1_f32, 0.2, 0.3, 0.4], &[4]).expect("falha ao criar FlowTensors probs");
    let samples = FlowTensors::multinomial(&probs, 4).expect("falha ao executar multinomial");
    println!("Multinomial samples (indices): {:?}", samples.data().expect("falha ao obter dados de samples"));
    
    // ==================== FUNÇÕES DE ATIVAÇÃO ====================
    println!("\n3. FUNÇÕES DE ATIVAÇÃO");
    println!("{}", "-".repeat(50));
    
    let activations = Tensor::from_values(&[-2.0, -1.0, 0.0, 1.0, 2.0], 1, 5);
    println!("Valores originais:");
    activations.print();
    
    let relu = activations.relu();
    println!("\nReLU:");
    relu.print();
    
    let sigmoid = activations.sigmoid();
    println!("\nSigmoid:");
    sigmoid.print();
    
    let tanh = activations.tanh();
    println!("\nTanh:");
    tanh.print();
    
    // ==================== ESTATÍSTICAS ====================
    println!("\n4. ESTATÍSTICAS AVANÇADAS");
    println!("{}", "-".repeat(50));
    
    let data = Tensor::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    println!("Dataset:");
    data.print();
    
    println!("\nEstatísticas:");
    println!("  Soma: {:.4}", data.sum());
    println!("  Média: {:.4}", data.mean());
    println!("  Desvio Padrão: {:.4}", data.std());
    println!("  Variância: {:.4}", data.var());
    println!("  Máximo: {:.4}", data.max());
    println!("  Mínimo: {:.4}", data.min());
    println!("  Argmax (índice): {}", data.argmax());
    println!("  Argmin (índice): {}", data.argmin());
    
    // ==================== NEURAL NETWORK COM NOVAS FEATURES ====================
    println!("\n5. REDE NEURAL COM NOVAS FEATURES");
    println!("{}", "-".repeat(50));
    
    // Criar uma rede neural simples
    let layer1 = Linear::new(2, 3);
    let layer2 = Linear::new(3, 1);
    
    // Otimizador Adam (NOVO!)
    let optimizer = Optimizer::adam(&layer1, 0.01);
    
    // Dados de treinamento
    let x_train = Tensor::from_values(&[0.5, 0.3, 0.8, 0.2, 0.1, 0.9], 3, 2);
    let y_train = Tensor::from_values(&[1.0, 0.5, 0.2], 3, 1);
    
    println!("Treinando com Adam optimizer...");
    
    for epoch in 0..5 {
        // Forward pass com ativação ReLU
        let h1 = layer1.forward(&x_train);
        let h1_activated = h1.relu(); // ReLU activation (NOVO!)
        
        let output = layer2.forward(&h1_activated);
        
        // Loss (pode usar MSE ou CrossEntropy)
        let loss = output.mse_loss(&y_train);
        
        println!("Epoch {}: Loss = {:.6}", epoch, loss.as_slice()[0]);
        
        // Backward e otimização
        loss.backward();
        optimizer.step();
    }
}
