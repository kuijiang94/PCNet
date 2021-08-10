def shuffle_down(inputs, scale):
    N, C, iH, iW = inputs.size()
    oH = iH // scale
    oW = iW // scale

    output = inputs.view(N, C, oH, scale, oW, scale)
    output = output.permute(0,1,5,3,2,4).contiguous()
    return output.view(N, -1, oH, oW)


def shuffle_up(inputs, scale):
    N, C, iH, iW = inputs.size()
    oH = iH * scale
    oW = iW * scale
    oC = C // (scale ** 2)
    output = inputs.view(N, oC, scale, scale, iH, iW)
    output = output.permute(0,1,4,3,5,2).contiguous()
    output = output.view(N, oC, oH, oW)
    return output

