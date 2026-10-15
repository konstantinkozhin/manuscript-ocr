import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from manuscript.detectors._east.sam import SAMSolver
TORCH_AVAILABLE = True


# --- Tests for SAMSolver initialization ---
def test_sam_solver_initialization_default():
    """Test SAMSolver initialization with default parameters"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    assert optimizer.use_adaptive == False
    assert len(optimizer.param_groups) > 0
    assert optimizer.param_groups[0]["rho"] == 0.05
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_sam_solver_initialization_custom_rho():
    """Test SAMSolver initialization with custom rho"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, rho=0.1, lr=0.01
    )

    assert optimizer.param_groups[0]["rho"] == 0.1


def test_sam_solver_initialization_adaptive():
    """Test SAMSolver initialization with adaptive mode"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        use_adaptive=True,
        lr=0.01,
    )

    assert optimizer.use_adaptive == True


def test_sam_solver_initialization_negative_rho():
    """Test SAMSolver initialization with negative rho"""
    model = nn.Linear(10, 5)

    with pytest.raises(ValueError, match="rho must be non-negative"):
        SAMSolver(
            model.parameters(), base_optimizer_cls=torch.optim.SGD, rho=-0.1, lr=0.01
        )


def test_sam_solver_initialization_adam():
    """Test SAMSolver initialization with Adam optimizer"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001
    )

    assert isinstance(optimizer._optimizer, torch.optim.Adam)


def test_sam_solver_initialization_different_optimizers():
    """Test SAMSolver initialization with different optimizers"""
    model = nn.Linear(10, 5)

    # SGD
    opt_sgd = SAMSolver(model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01)
    assert isinstance(opt_sgd._optimizer, torch.optim.SGD)

    # Adam
    opt_adam = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001
    )
    assert isinstance(opt_adam._optimizer, torch.optim.Adam)

    # RMSprop
    opt_rmsprop = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.RMSprop, lr=0.01
    )
    assert isinstance(opt_rmsprop._optimizer, torch.optim.RMSprop)


# --- Tests for step method ---
def test_sam_solver_step_basic():
    """Test basic SAMSolver optimization step"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    initial_params = [p.clone() for p in model.parameters()]
    loss = optimizer.step(closure)

    # Check that parameters changed
    params_changed = any(
        not torch.equal(p1, p2) for p1, p2 in zip(initial_params, model.parameters())
    )
    assert params_changed
    assert isinstance(loss, torch.Tensor)


def test_sam_solver_step_reduces_loss():
    """Test that SAMSolver reduces loss"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1
    )

    x = torch.randn(10, 10)
    y = torch.randn(10, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    initial_loss = closure().item()

    # Multiple optimization steps
    for _ in range(10):
        optimizer.step(closure)

    final_loss = closure().item()

    # Loss should decrease after several iterations
    assert final_loss < initial_loss


def test_sam_solver_step_with_adaptive():
    """Test SAMSolver step in adaptive mode"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        use_adaptive=True,
        lr=0.01,
    )

    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    loss = optimizer.step(closure)
    assert isinstance(loss, torch.Tensor)


def test_sam_solver_step_multiple_iterations():
    """Test multiple SAMSolver iterations"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    x = torch.randn(5, 10)
    y = torch.randn(5, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    losses = []
    for _ in range(5):
        loss = optimizer.step(closure)
        losses.append(loss.item())

    # All losses should be valid
    assert all(l >= 0 for l in losses)


# --- Tests for zero_grad ---
def test_sam_solver_zero_grad():
    """Test zero_grad method"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    # Create gradients
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()

    # Check that gradients exist
    assert any(p.grad is not None for p in model.parameters())

    # Zero gradients
    optimizer.zero_grad()

    # Check that gradients are zeroed
    assert all(
        p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
        for p in model.parameters()
    )


# --- Tests for state_dict and load_state_dict ---
def test_sam_solver_state_dict():
    """Test getting state_dict"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    state = optimizer.state_dict()

    assert isinstance(state, dict)
    assert "state" in state or "param_groups" in state


def test_sam_solver_load_state_dict():
    """Test loading state_dict"""
    model = nn.Linear(10, 5)
    optimizer1 = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    # Perform an optimization step
    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    def closure():
        optimizer1.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    optimizer1.step(closure)

    # Save state
    state = optimizer1.state_dict()

    # Create new optimizer and load state
    model2 = nn.Linear(10, 5)
    optimizer2 = SAMSolver(
        model2.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )
    optimizer2.load_state_dict(state)

    # States should be identical
    assert optimizer2.state_dict().keys() == optimizer1.state_dict().keys()


def test_sam_solver_state_persistence():
    """Test saving and restoring optimizer state"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001
    )

    x = torch.randn(5, 10)
    y = torch.randn(5, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    # Make several steps
    for _ in range(3):
        optimizer.step(closure)

    # Save state
    saved_state = optimizer.state_dict()

    # Make more steps
    for _ in range(3):
        optimizer.step(closure)

    # Restore state
    optimizer.load_state_dict(saved_state)

    # State should be restored (check structure)
    current_state = optimizer.state_dict()
    assert current_state.keys() == saved_state.keys()
    assert len(current_state["param_groups"]) == len(saved_state["param_groups"])


# --- Tests for _compute_grad_magnitude ---
def test_sam_solver_compute_grad_magnitude_with_gradients():
    """Test computing grad magnitude with gradients"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    # Create gradients
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()

    # Compute grad magnitude
    grad_norm = optimizer._compute_grad_magnitude()

    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm > 0


def test_sam_solver_compute_grad_magnitude_no_gradients():
    """Test computing grad magnitude without gradients"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    # No gradients
    grad_norm = optimizer._compute_grad_magnitude()

    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm == 0.0


def test_sam_solver_compute_grad_magnitude_adaptive():
    """Test computing grad magnitude in adaptive mode"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        use_adaptive=True,
        lr=0.01,
    )

    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()

    grad_norm = optimizer._compute_grad_magnitude()

    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm > 0


# --- Integration tests ---
def test_sam_solver_training_loop():
    """Test full training loop with SAMSolver"""
    # Simple regression task
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    # Generate data
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)

    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        return loss

    initial_loss = closure().item()

    # Train
    for _ in range(20):
        loss = optimizer.step(closure)

    final_loss = loss.item()

    # Loss should decrease
    assert final_loss < initial_loss


def test_sam_solver_different_model_architectures():
    """Test SAMSolver with different model architectures"""
    x = torch.randn(5, 10)
    y = torch.randn(5, 3)

    # Simple linear model
    model1 = nn.Linear(10, 3)
    opt1 = SAMSolver(model1.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01)

    def closure1():
        opt1.zero_grad()
        loss = nn.functional.mse_loss(model1(x), y)
        return loss

    loss1 = opt1.step(closure1)
    assert torch.isfinite(loss1)

    # Multi-layer model
    model2 = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3)
    )
    opt2 = SAMSolver(model2.parameters(), base_optimizer_cls=torch.optim.Adam, lr=0.001)

    def closure2():
        opt2.zero_grad()
        loss = nn.functional.mse_loss(model2(x), y)
        return loss

    loss2 = opt2.step(closure2)
    assert torch.isfinite(loss2)


def test_sam_solver_batch_training():
    """Test SAMSolver with batch training"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    # Different batch sizes
    for batch_size in [1, 5, 10, 20]:
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 5)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            return loss

        loss = optimizer.step(closure)
        assert torch.isfinite(loss)


def test_sam_solver_rho_values():
    """Test SAMSolver with different rho values"""
    model = nn.Linear(10, 5)
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)

    for rho in [0.01, 0.05, 0.1, 0.5]:
        optimizer = SAMSolver(
            model.parameters(), base_optimizer_cls=torch.optim.SGD, rho=rho, lr=0.01
        )

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            return loss

        loss = optimizer.step(closure)
        assert torch.isfinite(loss)


def test_sam_solver_cuda_device():
    """Test SAMSolver on CUDA device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = nn.Linear(10, 5).cuda()
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    x = torch.randn(3, 10).cuda()
    y = torch.randn(3, 5).cuda()

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    loss = optimizer.step(closure)

    assert loss.device.type == "cuda"


def test_sam_solver_cpu_device():
    """Test SAMSolver on CPU device"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    x = torch.randn(3, 10)
    y = torch.randn(3, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    loss = optimizer.step(closure)

    assert loss.device.type == "cpu"


def test_sam_solver_parameter_updates():
    """Test that parameters are updated correctly"""
    model = nn.Linear(5, 3)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.1
    )

    x = torch.randn(10, 5)
    y = torch.randn(10, 3)

    # Save initial parameters
    initial_weight = model.weight.clone()
    initial_bias = model.bias.clone()

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    # Make a step
    optimizer.step(closure)

    # Parameters should change
    assert not torch.equal(model.weight, initial_weight)
    assert not torch.equal(model.bias, initial_bias)


def test_sam_solver_convergence():
    """Test SAMSolver convergence on simple task"""
    torch.manual_seed(123)

    # Create simple linear dependency
    true_w = torch.randn(5, 1)
    true_b = torch.randn(1)

    X = torch.randn(100, 5)
    y = X @ true_w + true_b + torch.randn(100, 1) * 0.1

    model = nn.Linear(5, 1)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01
    )

    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        return loss

    losses = []
    for _ in range(50):
        loss = optimizer.step(closure)
        losses.append(loss.item())

    # Check that loss generally decreases
    # Take average of first 10 and last 10
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10

    assert late_avg < early_avg


def test_sam_solver_with_momentum():
    """Test SAMSolver with SGD + momentum"""
    model = nn.Linear(10, 5)
    optimizer = SAMSolver(
        model.parameters(),
        base_optimizer_cls=torch.optim.SGD,
        lr=0.01,
        momentum=0.9,
    )

    x = torch.randn(5, 10)
    y = torch.randn(5, 5)

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    loss = optimizer.step(closure)
    assert torch.isfinite(loss)


def test_sam_solver_small_model():
    """Test SAMSolver with small model"""
    model = nn.Linear(2, 1)
    optimizer = SAMSolver(
        model.parameters(), base_optimizer_cls=torch.optim.SGD, lr=0.01  # Reduced lr
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[3.0], [7.0]])

    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        return loss

    initial_loss = closure().item()

    for _ in range(10):
        loss = optimizer.step(closure)

    final_loss = loss.item()

    # Check that loss is finite and didn't explode
    assert torch.isfinite(torch.tensor(final_loss))
