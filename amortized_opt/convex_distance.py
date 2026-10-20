"""Batched capsule/capped-cylinder signed distance using support planes.

For convex bodies the maximum separating-plane gap is Euclidean distance
outside contact and negative minimum translational penetration inside contact.
This is the signed support-function query underlying GJK/EPA, specialized to
these two primitives. No meshes, CPU transfers, or differentiation through an
iterative collision solver are needed.

A maximizing normal is either a cylinder cap normal, an endpoint-to-cylinder
normal, or perpendicular to the capsule axis. In the latter case the remaining
one-dimensional support problem is solved by bisection on its concave branch,
including the endpoints. The normal is held fixed in backward (the envelope
theorem); gradients flow through the final support gap. Feature changes are
nonsmooth, just like an exact closest-feature distance query.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _scalar_like(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype).expand(reference.shape[:-1])
    return reference.new_full(reference.shape[:-1], float(value))


def _perpendicular(axis):
    # Choose the coordinate direction least aligned with the axis.
    index = axis.abs().argmin(-1)
    seed = F.one_hot(index, 3).to(axis.dtype)
    return F.normalize(torch.linalg.cross(axis, seed), dim=-1)


def capsule_cylinder_signed_distance(a0, a1, center, axis, radius, half_height,
                                     capsule_radius=0.0, *, ra=None):
    """Signed distance in metres, broadcasting all leading dimensions.

    Positive = separation; negative = minimum translational penetration.
    The cylinder axis must be nonzero; all radii/heights are nonnegative.
    Bisection uses 32 iterations on detached geometry; the final query remains
    differentiable in endpoints, cylinder pose, radius, height and capsule
    radius. Internal arithmetic is at least float32.
    """
    if ra is not None:
        capsule_radius = ra
    a0, a1, center, axis = torch.broadcast_tensors(a0, a1, center, axis)
    dtype = a0.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    a0, a1, center, axis = (x.to(work_dtype) for x in (a0, a1, center, axis))
    axis = F.normalize(axis, dim=-1)
    radius = _scalar_like(radius, a0)
    height = _scalar_like(half_height, a0)
    capsule_radius = _scalar_like(capsule_radius, a0)
    eps = 1e-14 if work_dtype == torch.float64 else 1e-7

    with torch.no_grad():
        p0, p1, c, k, R, H = (x.detach() for x in (a0, a1, center, axis, radius, height))
        segment = p1-p0
        length = torch.linalg.vector_norm(segment, dim=-1, keepdim=True)
        u = torch.where(length > eps, segment/length.clamp_min(eps), k)
        radial_fallback = _perpendicular(k)
        endpoint = torch.stack((p0-c, p1-c), dim=-2)
        z = (endpoint*k[..., None, :]).sum(-1)
        radial = endpoint-z[..., None]*k[..., None, :]
        rho = torch.linalg.vector_norm(radial, dim=-1)
        radial_normal = torch.where((rho > eps)[..., None],
            radial/rho.clamp_min(eps)[..., None], radial_fallback[..., None, :])
        axial_normal = torch.where((z >= 0)[..., None], k[..., None, :], -k[..., None, :])
        dr = rho-R[..., None]
        dz = z.abs()-H[..., None]
        outward = dr.clamp_min(0)[..., None]*radial_normal + dz.clamp_min(0)[..., None]*axial_normal
        inside_normal = torch.where((dr > dz)[..., None], radial_normal, axial_normal)
        endpoint_normal = torch.where(((dr > 0) | (dz > 0))[..., None],
            F.normalize(outward, dim=-1), inside_normal)

        # Interior capsule support has n.u = 0. Parameterize that great circle
        # with e1 along the projected cylinder axis and e2 orthogonal to it.
        projected = k-(k*u).sum(-1, keepdim=True)*u
        kappa = torch.linalg.vector_norm(projected, dim=-1).clamp(0, 1)
        e1 = torch.where((kappa > eps)[..., None], projected/kappa.clamp_min(eps)[..., None],
                         _perpendicular(u))
        e2 = torch.linalg.cross(u, e1)
        delta = (p0+p1)*0.5-c
        dx = (delta*e1).sum(-1)
        dy = (delta*e2).sum(-1)
        e2 = torch.where((dy >= 0)[..., None], e2, -e2)
        B = dy.abs()
        K = kappa.square()
        # On either half-circle, maximize
        # A*x + B*sqrt(1-x*x) - R*sqrt(1-K*x*x), 0<=x<=1.
        # Its curvature changes sign at most once, from convex to concave.
        ratio = (B/(R*K).clamp_min(eps)).clamp(0, 1).pow(2/3)
        turn = torch.sqrt(((1-ratio)/(1-ratio*K).clamp_min(eps)).clamp(0, 1))
        signs = torch.stack((-torch.ones_like(dx), torch.ones_like(dx)), dim=-1)
        A = dx[..., None]*signs-H[..., None]*kappa[..., None]
        lo = turn[..., None].expand_as(A).clone()
        hi = torch.ones_like(lo)
        for _ in range(32):
            x = (lo+hi)*0.5
            derivative = (A-B[..., None]*x/torch.sqrt((1-x*x).clamp_min(eps))
                + (R*K)[..., None]*x/torch.sqrt((1-K[..., None]*x*x).clamp_min(eps)))
            lo = torch.where(derivative > 0, x, lo)
            hi = torch.where(derivative > 0, hi, x)
        x = (lo+hi)*0.5
        interior = (signs*x)[..., None]*e1[..., None, :] + torch.sqrt((1-x*x).clamp_min(0))[..., None]*e2[..., None, :]
        candidates = torch.cat((k[..., None, :], -k[..., None, :], endpoint_normal,
            e1[..., None, :], -e1[..., None, :], e2[..., None, :], interior), dim=-2)
        candidates = F.normalize(candidates, dim=-1)
        gap = torch.minimum(((p0-c)[..., None, :]*candidates).sum(-1),
                            ((p1-c)[..., None, :]*candidates).sum(-1))
        axial = (candidates*k[..., None, :]).sum(-1)
        tangent = candidates-axial[..., None]*k[..., None, :]
        gap = gap-H[..., None]*axial.abs()-R[..., None]*torch.linalg.vector_norm(tangent, dim=-1)
        index = gap.argmax(-1)
        normal = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
        # At an interior capsule contact both endpoints support the same
        # plane. Differentiating min(endpoint projections) would arbitrarily
        # put all force on one endpoint. Recover the actual segment witness
        # instead, including cap/side nonunique cylinder witnesses.
        nk = (normal*k).sum(-1)
        nr = normal-nk[..., None]*k
        nr_length = torch.linalg.vector_norm(nr, dim=-1)
        cylinder_witness = (c + torch.where((nk >= 0)[..., None], k, -k)*H[..., None]
                            + R[..., None]*nr/nr_length.clamp_min(eps)[..., None])
        t_rim = ((cylinder_witness-p0)*segment).sum(-1)/length.squeeze(-1).square().clamp_min(eps)
        p_radial = p0-c-((p0-c)*k).sum(-1, keepdim=True)*k
        s_radial = segment-(segment*k).sum(-1, keepdim=True)*k
        t_flat = -(p_radial*s_radial).sum(-1)/s_radial.square().sum(-1).clamp_min(eps)
        t_interior = torch.where((nk.abs() < 1e-6) | (nr_length < 1e-6), t_flat, t_rim).clamp(0, 1)
        projection = (normal*segment).sum(-1)
        t = torch.where(projection.abs() < 1e-6*length.squeeze(-1), t_interior,
                        (projection < 0).to(projection.dtype))
        point = p0+segment*t[..., None]-c
        cylinder_z = torch.where(nk.abs() < 1e-6, (point*k).sum(-1).clamp(-H, H),
                                 torch.where(nk >= 0, H, -H))
        cylinder_radial = R[..., None]*nr/nr_length.clamp_min(eps)[..., None]
        cylinder_radial = torch.where((nr_length < 1e-6)[..., None],
                                      point-(point*k).sum(-1, keepdim=True)*k,
                                      cylinder_radial)
        axis_gradient = -cylinder_z[..., None]*nr + nk[..., None]*cylinder_radial

    # Re-evaluate only the selected support plane with live geometry.
    witness = a0+(a1-a0)*t[..., None]
    witness_gap = ((witness-center)*normal).sum(-1)
    support_gap = torch.minimum(((a0-center)*normal).sum(-1), ((a1-center)*normal).sum(-1))
    # Keep the forward query a true support-plane lower bound even when
    # finite-precision normals make tied endpoint projections slightly differ.
    # The actual recovered witness supplies the envelope derivative.
    gap = support_gap.detach()+(witness_gap-witness_gap.detach())
    # Support at cap/side normals has nonunique cylinder witnesses. Use the
    # recovered witness's rotational derivative rather than norm'(0)=0.
    orientation = ((axis-axis.detach())*axis_gradient).sum(-1)
    return (gap-height*nk.abs()-radius*nr_length-capsule_radius+orientation).to(dtype)
