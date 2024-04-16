import torch

def calculate_percentage(lst):
    total_elements = len(lst)

    if total_elements == 0:
        return {}

    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for num in lst:
        counts[num] += 1

    percentages = {key: (count / total_elements) * 100 for key, count in counts.items()}

    return percentages

def normalize_reward(reward):
    min_reward = -50
    max_reward = 50

    # Ensure max_reward and min_reward are different to avoid division by zero
    if min_reward == max_reward:
        raise ValueError("min_reward and max_reward must be different values.")
    
    # Normalize the reward to the range [-1, 1]
    normalized_reward = 2 * (reward - min_reward) / (max_reward - min_reward) - 1
    
    # Clip the normalized reward to ensure it is within [-1, 1]
    normalized_reward = max(-1, min(1, normalized_reward))
    
    return normalized_reward

def custom_backward(action_probs, chosen_action, reward, optimizer):
    reward = normalize_reward(reward)

    # Calculate the loss as the negative log probability of the chosen action

    epsilon = 1e-8  # or a small value suitable for your problem

    # Add epsilon to prevent taking the log of zero
    selected_probs = torch.clamp(action_probs[0, chosen_action], epsilon, 1.0)

    # print("Action prob:", selected_probs)
    # print("Action prob -log:", -torch.log(selected_probs))
    # print("Reward:", reward)

    loss = -torch.log(selected_probs) * reward

    # print("Loss:", loss)

    # Take the sum to get a scalar loss
    # loss = loss.sum()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update the model parameters
    optimizer.step()

    return loss.item()