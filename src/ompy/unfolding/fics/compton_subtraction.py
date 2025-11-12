
def compton_subtraction(
    res: UnfoldedResult1D | UnfoldedResult2D,
    response: Response,
    space="eta",
    use_eff: bool = False,
) -> Vector | Matrix:
    if space == "eta":
        u = res.best_eta()
    elif space == "mu":
        u = res.best()
    else:
        raise ValueError(f"Invalid space: {space}")
    return compton_subtraction_(u, res.raw, response, use_eff=use_eff)


def compton_subtraction_(
    unfolded: Vector, raw: Vector, response: Response, use_eff: bool = False
):
    G = response.gaussian_like(unfolded).T
    eff = response.interpolation.Eff(unfolded.observed)

    f = response.fold_componentwise(unfolded)
    fe, se, de, ap, compton0 = f.FE, f.SE, f.DE, f.AP, f.compton
    pfe = response.component_matrices_like(unfolded)["FE"].sum("true")
    # Need to smooth AP to correct for commutator
    ap = G @ ap
    ap *= 0

    # The discrete structures: w
    w = se + de + ap
    # Total, without compton: v
    v = fe + w
    # Assume everything left over from the raw spectrum is the compton.
    # Incredible bad assumption, doesn't take into account the noise
    # of the raw spectrum, nor, more importantly, the errors made in the unfolding.
    # More succinctly: the peaks FE, SE, DE, AP, can only be assumed to be correct
    # when the unfolding is correct, but if the unfolding were correct, there
    # would be no need for the compton subtraction method to be used!
    compton = raw - v
    ax, _ = compton0.plot(label="compton 0")
    compton.plot(ax=ax, label="compton 1")
    # We know the compton is smooth, so smooth it.
    # Assume this is to correct for the noise, but this is too
    # ad-hoc.
    compton = G @ compton
    compton.plot(ax=ax, label="compton 2")
    ax.legend()

    # The raw spectrum minus the modeled compton, and the folded discrete structures
    # is the unfolded spectrum. I don't like this either, since now the
    # noise of the raw spectrum infects the unfolded spectrum, which we *also*
    # know must be smooth.
    unf = (raw - compton - w) / pfe

    if use_eff:
        unf = unf / eff

    ax0, _ = unf.plot(label="unf")
    compton.plot(ax=ax0, label="compton")
    fe.plot(ax=ax0, label="fe")
    se.plot(ax=ax0, label="se")
    de.plot(ax=ax0, label="de")
    ap.plot(ax=ax0, label="uap")
    raw.plot(ax=ax0, label="raw")
    v.plot(ax=ax0, label="v")
    unfolded.plot(ax=ax0, label="unfolded")
    ax0.legend()

    return ax, ax0



