import torch


def bias_regularized_margin_ranking_loss(input1, input2, regularizer, bias):
    """
    implementation of the a regularized version of the multilable margine loss, which is used in the
    repbert paper.

    1-[\SUM REL(Q,D_J^+)-(REL(Q,D_J^-)+LAMBDA BIAS(D_J^-))]
    BIAS(D_J^-) = ABS(boolean bias)
    Args:
        input1: rel(q, d+)
        input2: rel(q, d-)
        regularizer: scalar
        bias: bias of batch
    Returns:

    """
    diff = input2 + regularizer * bias - input1 # input2 - input1 to cover the minus sign of target in the original formula
    input_loss = diff + torch.ones(input1.size()).to('cuda')
    max_0 = torch.nn.functional.relu(input_loss)
    return torch.mean(max_0)


def refinement_regularized_margin_ranking_loss(score_positive, score_negative, score_queries):
    """
    implementation of the a regularized version of the multilable margine loss, which is used in the
    repbert paper.

    max(0, 1-[REL(Q,D_J^+)-(REL(Q,D_J^-)+ REL(Q, Q'))])

    Args:
        input1:

    Returns:

    """
    diff = score_negative - score_positive - score_queries# input2 - input1 to cover the minus sign of target in the original formula
    input_loss = diff + torch.ones(score_positive.size()).to('cuda')
    max_0 = torch.nn.functional.relu(input_loss)
    return torch.mean(max_0)
