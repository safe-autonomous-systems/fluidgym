from __future__ import annotations

import collections.abc
import typing

import torch

__all__: list[str] = [
    "ABS_MAX",
    "ABS_MEAN",
    "ABS_SUM",
    "AdvectionScheme",
    "Block",
    "Boundary",
    "BoundaryConditionType",
    "BoundarySampling",
    "BoundaryType",
    "CENTRAL",
    "CLAMP",
    "CONNECTED",
    "CONSTANT",
    "CSRmatrix",
    "ComputeCurrentDensityFaceBased",
    "ComputeCurrentDensityFaceBasedGrad",
    "ComputeEpotRHS",
    "ComputeEpotRHSGrad",
    "ComputeFieldGradient",
    "ComputeFieldGradientFVM",
    "ComputePressureGradient",
    "ComputeSpatialVelocityGradients",
    "ComputeVelocityDivergence",
    "ConnectedBoundary",
    "ConvergenceCriterion",
    "CoordsToFaceTransforms",
    "CoordsToTransforms",
    "CopyEpotResultFromBlocks",
    "CopyEpotResultGradFromBlocks",
    "CopyEpotResultToBlocks",
    "CopyPressureResultFromBlocks",
    "CopyPressureResultGradFromBlocks",
    "CopyPressureResultToBlocks",
    "CopyScalarResultFromBlocks",
    "CopyScalarResultGradFromBlocks",
    "CopyScalarResultGradToBlocks",
    "CopyScalarResultToBlocks",
    "CopyVelocityResultFromBlocks",
    "CopyVelocityResultGradFromBlocks",
    "CopyVelocityResultGradToBlocks",
    "CopyVelocityResultToBlocks",
    "CorrectVelocity",
    "CorrectVelocityGrad",
    "DIRICHLET",
    "DIRICHLET_VARYING",
    "Domain",
    "EigenDecomposition",
    "FIXED",
    "FixedBoundary",
    "Int4",
    "InvertMatrix",
    "LINEAR_UPWIND",
    "LinearSolverResultInfo",
    "MakeBasisUnique",
    "MakeCoordsNDNonUniformScaleNormalized",
    "MakeGrid2DNonUniformScale",
    "MakeGridNDExpScaleNormalized",
    "MakeGridNDNonUniformScaleNormalized",
    "NEUMANN",
    "NORM2",
    "NORM2_NORMALIZED",
    "PERIODIC",
    "PeriodicBoundary",
    "SGSviscosityIncompressibleSmagorinsky",
    "SGSviscosityIncompressibleWALE",
    "SampleTransformedGridGlobalToLocal",
    "SampleTransformedGridLocalToGlobal",
    "SampleTransformedGridLocalToGlobalMulti",
    "SetupAdvectionMatrix",
    "SetupAdvectionMatrixGrad",
    "SetupAdvectionScalar",
    "SetupAdvectionScalarGrad",
    "SetupAdvectionVelocity",
    "SetupAdvectionVelocityGrad",
    "SetupEpotMatrix",
    "SetupPressureCorrection",
    "SetupPressureCorrectionGrad",
    "SetupPressureMatrix",
    "SetupPressureMatrixGrad",
    "SetupPressureRHS",
    "SetupPressureRHSGrad",
    "SetupPressureRHSdiv",
    "SetupPressureRHSdivGrad",
    "SolveLinear",
    "SparseOuterProduct",
    "StaticDirichletBoundary",
    "TransformVectors",
    "VaryingDirichletBoundary",
    "VectorToDiagMatrix",
    "matmul",
    "matmulGrad",
]

class AdvectionScheme:
    """
    Members:

      CENTRAL

      LINEAR_UPWIND
    """

    CENTRAL: typing.ClassVar[AdvectionScheme]  # value = <AdvectionScheme.CENTRAL: 0>
    LINEAR_UPWIND: typing.ClassVar[
        AdvectionScheme
    ]  # value = <AdvectionScheme.LINEAR_UPWIND: 1>
    __members__: typing.ClassVar[
        dict[str, AdvectionScheme]
    ]  # value = {'CENTRAL': <AdvectionScheme.CENTRAL: 0>, 'LINEAR_UPWIND': <AdvectionScheme.LINEAR_UPWIND: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Block:
    def CloseAllBoundaries(self) -> None:
        """
        Create FixedBoundary with Dirichlet-0 velocity and scalar at all faces. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.
        """
    @typing.overload
    def CloseBoundary(
        self,
        faceIndex: typing.SupportsInt,
        velocity: torch.Tensor | None = None,
        passiveScalar: torch.Tensor | None = None,
    ) -> None:
        """
        Create a FixedBoundary with Dirichlet velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.
        """
    @typing.overload
    def CloseBoundary(
        self,
        faceString: str,
        velocity: torch.Tensor | None = None,
        passiveScalar: torch.Tensor | None = None,
    ) -> None:
        """
        Create a FixedBoundary with Dirichlet velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.
        """
    @typing.overload
    def CloseBoundary_Old(self, faceIndex: typing.SupportsInt) -> None:
        """
        Create a DirichletBoundary with Dirichlet-0 velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.
        """
    @typing.overload
    def CloseBoundary_Old(self, faceString: str) -> None:
        """
        Create a DirichletBoundary with Dirichlet-0 velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.
        """
    def ComputeCSRSize(self) -> int: ...
    @typing.overload
    def ConnectBlock(
        self,
        faceIndex: typing.SupportsInt,
        otherBlock: Block,
        otherFaceIndex: typing.SupportsInt,
        axis1Index: typing.SupportsInt = -1,
        axis2Index: typing.SupportsInt = -1,
    ) -> None:
        """
        Make a connection between this block and another.directional face specification: [-x,+x,-y,+y,-z,+z] <-> [0,5]=> axis := face/2 ([x,y,z] <-> [0,2])=> direction := face%2, 0 is lower/negative side, 1 is upper/positive side	for 'axisIndex' the direction indicates if the connection is inverted (0 for same direction, 1 for inverted)faceIndex of the block is connected to otherFaceIndex of otherBlock.for 2D and 3D, the remaining axes are also mapped:	faceIndex connects to otherFaceIndex	axis[(faceIndex / 2 + 1)%dims] is aligned to axis1Index. The connection is inverted if axis1Index%2==1.	axis[(faceIndex / 2 + 2)%dims] is aligned to axis2Index. The connection is inverted if axis2Index%2==1.
        """
    @typing.overload
    def ConnectBlock(
        self,
        faceString: str,
        otherBlock: Block,
        otherFaceString: str,
        axis1String: str = "",
        axis2String: str = "",
    ) -> None:
        """
        Make a connection between this block and another
        """
    def CreateEpot(self) -> None: ...
    def CreateEpotGrad(self) -> None: ...
    def CreatePassiveScalar(self) -> None: ...
    def CreatePassiveScalarGrad(self) -> None: ...
    def CreatePressure(self) -> None: ...
    def CreatePressureGrad(self) -> None: ...
    def CreateVelocity(self) -> None: ...
    def CreateVelocityGrad(self) -> None: ...
    def CreateVelocitySource(self, arg0: bool) -> None: ...
    def CreateVelocitySourceGrad(self) -> None:
        """
        Only create grad if velocity source exists, clears it otherwise
        """
    def CreateViscosityGrad(self) -> None: ...
    def Detach(self) -> None: ...
    def DetachFwd(self) -> None: ...
    def DetachGrad(self) -> None: ...
    def GetCoordinateOrientation(self) -> int:
        """
        Check if all transformed coordinate systems have the same orientation/handedness by checking the sign of the determinant.Returns the sign if all have the same orientation, 0 otherwise.Returns 1 if no transformations are set.
        """
    def IsTensorChanged(self) -> bool: ...
    def MakeAllPeriodic(self) -> None:
        """
        Make the block perioic along all axes
        """
    @typing.overload
    def MakePeriodic(self, axisIndex: typing.SupportsInt) -> None:
        """
        Make the block periodic along the given logical axis
        """
    @typing.overload
    def MakePeriodic(self, axisString: str) -> None:
        """
        Make the block periodic along the given logical axis
        """
    @typing.overload
    def OpenBoundary(
        self, faceIndex: typing.SupportsInt, passiveScalar: torch.Tensor | None = None
    ) -> None:
        """
        Create a FixedBoundary with zero-Neumann (free-slip) velocity BC at the specified face. Normal velocity is zero; tangential velocity has zero gradient (du/dn=0). Passive scalar defaults to Neumann unless a tensor is provided (Dirichlet).
        """
    @typing.overload
    def OpenBoundary(
        self, faceString: str, passiveScalar: torch.Tensor | None = None
    ) -> None:
        """
        Create a FixedBoundary with zero-Neumann (free-slip) velocity BC at the specified face. Normal velocity is zero; tangential velocity has zero gradient (du/dn=0). Passive scalar defaults to Neumann unless a tensor is provided (Dirichlet).
        """
    def __str__(self) -> str: ...
    @typing.overload
    def _setBoundary(self, faceIndex: typing.SupportsInt, boundary: Boundary) -> None:
        """
        Directly set a boundary to a specific face. Please use the boundary factories below to create consistent boundaries.
        """
    @typing.overload
    def _setBoundary(self, faceString: str, boundary: Boundary) -> None:
        """
        Directly set a boundary to a specific face. Please use the boundary factories below to create consistent boundaries.
        """
    def clearCoordsTransforms(self) -> None:
        """
        Clears any set coordinates or transformations.
        """
    def clearVelocitySource(self) -> None: ...
    def clearVelocitySourceGrad(self) -> None: ...
    def clearViscosity(self) -> None: ...
    def clearViscosityGrad(self) -> None: ...
    def getBoundaries(self) -> list[Boundary]: ...
    @typing.overload
    def getBoundary(self, faceIndex: typing.SupportsInt) -> Boundary: ...
    @typing.overload
    def getBoundary(self, faceString: str) -> Boundary: ...
    def getCellCoordinates(self) -> torch.Tensor: ...
    def getCellSizes(self) -> torch.Tensor:
        """
        Returns the block's cell sizes as NCDHW tensor (with C=1)
        """
    def getDevice(self) -> torch.device: ...
    def getDim(self, arg0: typing.SupportsInt) -> int: ...
    def getFixedBoundaries(self) -> list[tuple[int, FixedBoundary]]:
        """
        Returns a list of pairs (boundary index, boundary) containing only the fixed boundaries of this block.
        """
    def getMaxVelocity(
        self, withBounds: bool = True, computational: bool = False
    ) -> torch.Tensor: ...
    def getMaxVelocityMagnitude(
        self, withBounds: bool = True, computational: bool = False
    ) -> torch.Tensor: ...
    def getParentDomain(self) -> Domain: ...
    def getPassiveScalarChannels(self) -> int: ...
    def getSizes(self) -> Int4: ...
    def getSpatialDims(self) -> int: ...
    def getStrides(self) -> Int4: ...
    def getVelocity(self, computational: bool) -> torch.Tensor:
        """
        Returns the velocity, optionally transformed to compuatational space.
        """
    def hasEpot(self) -> bool: ...
    def hasEpotGrad(self) -> bool: ...
    def hasFaceTransform(self) -> bool: ...
    def hasPassiveScalar(self) -> bool: ...
    def hasPassiveScalarViscosity(self) -> bool: ...
    def hasPrescribedBoundary(self) -> bool: ...
    def hasTransform(self) -> bool: ...
    def hasVelocitySource(self) -> bool: ...
    def hasVelocitySourceGrad(self) -> bool: ...
    def hasVertexCoordinates(self) -> bool: ...
    def hasViscosity(self) -> bool: ...
    def hasViscosityGrad(self) -> bool: ...
    def isAllFixedBoundariesPassiveScalarTypeStatic(self) -> bool: ...
    def isViscosityStatic(self) -> bool: ...
    def setEpot(self, arg0: torch.Tensor) -> None: ...
    def setEpotGrad(self, arg0: torch.Tensor) -> None: ...
    def setPassiveScalar(self, arg0: torch.Tensor) -> None: ...
    def setPassiveScalarGrad(self, arg0: torch.Tensor) -> None: ...
    def setPressure(self, arg0: torch.Tensor) -> None: ...
    def setPressureGrad(self, arg0: torch.Tensor) -> None: ...
    def setTransform(
        self, transform: torch.Tensor, faceTransform: torch.Tensor | None = None
    ) -> None:
        """
        Set cell transformation metrics, face transformations optional. Clears any previous transformations or coordinates.
        """
    def setVelocity(self, arg0: torch.Tensor) -> None: ...
    def setVelocityGrad(self, arg0: torch.Tensor) -> None: ...
    def setVelocitySource(self, arg0: torch.Tensor) -> None: ...
    def setVelocitySourceGrad(self, arg0: torch.Tensor) -> None: ...
    def setVertexCoordinates(self, vertexCoordinates: torch.Tensor) -> None:
        """
        Set the cell vertex coordinates, automatically calculates cell and face transformations and updates existing StaticDirichletBoundaries.
        """
    def setViscosity(self, arg0: torch.Tensor) -> None: ...
    def setViscosityGrad(self, arg0: torch.Tensor) -> None: ...
    @property
    def csrOffset(self) -> int: ...
    @property
    def epot(self) -> torch.Tensor | None: ...
    @property
    def epotGrad(self) -> torch.Tensor | None: ...
    @property
    def faceTransform(self) -> torch.Tensor | None: ...
    @property
    def globalOffset(self) -> int: ...
    @property
    def isVelocitySourceStatic(self) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def passiveScalar(self) -> torch.Tensor | None: ...
    @property
    def passiveScalarGrad(self) -> torch.Tensor: ...
    @property
    def passiveScalarViscosity(self) -> torch.Tensor | None: ...
    @property
    def pressure(self) -> torch.Tensor: ...
    @property
    def pressureGrad(self) -> torch.Tensor: ...
    @property
    def transform(self) -> torch.Tensor | None: ...
    @property
    def velocity(self) -> torch.Tensor: ...
    @property
    def velocityGrad(self) -> torch.Tensor: ...
    @property
    def velocitySource(self) -> torch.Tensor | None: ...
    @property
    def velocitySourceGrad(self) -> torch.Tensor | None: ...
    @property
    def vertexCoordinates(self) -> torch.Tensor | None: ...
    @property
    def viscosity(self) -> torch.Tensor | None: ...
    @property
    def viscosityGrad(self) -> torch.Tensor | None: ...

class Boundary:
    def __str__(self) -> str: ...
    def getParentDomain(self) -> Domain: ...
    @property
    def type(self) -> BoundaryType: ...

class BoundaryConditionType:
    """
    Members:

      DIRICHLET

      NEUMANN
    """

    DIRICHLET: typing.ClassVar[
        BoundaryConditionType
    ]  # value = <BoundaryConditionType.DIRICHLET: 0>
    NEUMANN: typing.ClassVar[
        BoundaryConditionType
    ]  # value = <BoundaryConditionType.NEUMANN: 1>
    __members__: typing.ClassVar[
        dict[str, BoundaryConditionType]
    ]  # value = {'DIRICHLET': <BoundaryConditionType.DIRICHLET: 0>, 'NEUMANN': <BoundaryConditionType.NEUMANN: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class BoundarySampling:
    """
    Members:

      CONSTANT

      CLAMP
    """

    CLAMP: typing.ClassVar[BoundarySampling]  # value = <BoundarySampling.CLAMP: 1>
    CONSTANT: typing.ClassVar[
        BoundarySampling
    ]  # value = <BoundarySampling.CONSTANT: 0>
    __members__: typing.ClassVar[
        dict[str, BoundarySampling]
    ]  # value = {'CONSTANT': <BoundarySampling.CONSTANT: 0>, 'CLAMP': <BoundarySampling.CLAMP: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class BoundaryType:
    """
    Members:

      DIRICHLET

      DIRICHLET_VARYING

      NEUMANN

      FIXED

      CONNECTED

      PERIODIC
    """

    CONNECTED: typing.ClassVar[BoundaryType]  # value = <BoundaryType.CONNECTED: 20>
    DIRICHLET: typing.ClassVar[BoundaryType]  # value = <BoundaryType.DIRICHLET: 0>
    DIRICHLET_VARYING: typing.ClassVar[
        BoundaryType
    ]  # value = <BoundaryType.DIRICHLET_VARYING: 1>
    FIXED: typing.ClassVar[BoundaryType]  # value = <BoundaryType.FIXED: 30>
    NEUMANN: typing.ClassVar[BoundaryType]  # value = <BoundaryType.NEUMANN: 10>
    PERIODIC: typing.ClassVar[BoundaryType]  # value = <BoundaryType.PERIODIC: 21>
    __members__: typing.ClassVar[
        dict[str, BoundaryType]
    ]  # value = {'DIRICHLET': <BoundaryType.DIRICHLET: 0>, 'DIRICHLET_VARYING': <BoundaryType.DIRICHLET_VARYING: 1>, 'NEUMANN': <BoundaryType.NEUMANN: 10>, 'FIXED': <BoundaryType.FIXED: 30>, 'CONNECTED': <BoundaryType.CONNECTED: 20>, 'PERIODIC': <BoundaryType.PERIODIC: 21>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CSRmatrix:
    def CreateValue(self) -> None: ...
    def IsTensorChanged(self) -> bool: ...
    def WithZeroValue(self) -> CSRmatrix: ...
    @typing.overload
    def __init__(
        self, arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        arg0: typing.SupportsInt,
        arg1: typing.SupportsInt,
        arg2: typing.Any,
        arg3: torch.device,
    ) -> None: ...
    def __str__(self) -> str: ...
    def clone(self) -> CSRmatrix: ...
    def copy(self) -> CSRmatrix: ...
    def detach(self) -> None:
        """
        Detach value tensor gradient in-place.
        """
    def getDevice(self) -> torch.device: ...
    def getRows(self) -> int: ...
    def getSize(self) -> int: ...
    def setValue(self, arg0: torch.Tensor) -> None: ...
    def toType(self, arg0: typing.Any) -> CSRmatrix: ...
    @property
    def index(self) -> torch.Tensor: ...
    @property
    def row(self) -> torch.Tensor: ...
    @property
    def value(self) -> torch.Tensor: ...

class ConnectedBoundary(Boundary):
    def __str__(self) -> str: ...
    def getConnectedBlock(self) -> Block: ...
    @property
    def axes(self) -> list[int]: ...
    @property
    def type(self) -> BoundaryType: ...

class ConvergenceCriterion:
    """
    Members:

      NORM2

      NORM2_NORMALIZED

      ABS_SUM

      ABS_MEAN

      ABS_MAX
    """

    ABS_MAX: typing.ClassVar[
        ConvergenceCriterion
    ]  # value = <ConvergenceCriterion.ABS_MAX: 4>
    ABS_MEAN: typing.ClassVar[
        ConvergenceCriterion
    ]  # value = <ConvergenceCriterion.ABS_MEAN: 3>
    ABS_SUM: typing.ClassVar[
        ConvergenceCriterion
    ]  # value = <ConvergenceCriterion.ABS_SUM: 2>
    NORM2: typing.ClassVar[
        ConvergenceCriterion
    ]  # value = <ConvergenceCriterion.NORM2: 0>
    NORM2_NORMALIZED: typing.ClassVar[
        ConvergenceCriterion
    ]  # value = <ConvergenceCriterion.NORM2_NORMALIZED: 1>
    __members__: typing.ClassVar[
        dict[str, ConvergenceCriterion]
    ]  # value = {'NORM2': <ConvergenceCriterion.NORM2: 0>, 'NORM2_NORMALIZED': <ConvergenceCriterion.NORM2_NORMALIZED: 1>, 'ABS_SUM': <ConvergenceCriterion.ABS_SUM: 2>, 'ABS_MEAN': <ConvergenceCriterion.ABS_MEAN: 3>, 'ABS_MAX': <ConvergenceCriterion.ABS_MAX: 4>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Domain:
    def AddBlock(self, arg0: Block) -> None: ...
    def CheckBoundaryFluxBalance(self, eps: typing.SupportsFloat = 1e-05) -> bool: ...
    def Clone(self, newName: str | None = None) -> Domain:
        """
        Copy the domain structure (blocks, boundaries, connections) and clone (memory copy) the primary tensor references (velociity, pressure, etc.).It is necessary to call PrepareSolve() on the copy to create the secondary tensors (matrices, RHS, results, gradients).
        """
    def Copy(self, newName: str | None = None) -> Domain:
        """
        Copy the domain structure (blocks, boundaries, connections) while keeping the same primary tensor references (velociity, pressure, etc.) as the original.It is necessary to call PrepareSolve() on the copy to create the secondary tensors (matrices, RHS, results, gradients).
        """
    def CreateA(self) -> None: ...
    def CreateAGrad(self) -> None: ...
    def CreateBlock(
        self,
        velocity: torch.Tensor | None = None,
        pressure: torch.Tensor | None = None,
        passiveScalar: torch.Tensor | None = None,
        vertexCoordinates: torch.Tensor | None = None,
        name: str = "Block",
    ) -> Block:
        """
        Create a block on the domain, at least one field/tensor must be specified, the rest if initialzed with zero if required.
        """
    def CreateBlockWithSize(self, size: Int4, name: str = "Block") -> Block:
        """
        Create a zero-initialized block on the domain with the given spatial size.
        """
    def CreateEpotGradOnBlocks(self) -> None: ...
    def CreateEpotMatrix(self) -> None: ...
    def CreateEpotOnBlocks(self) -> None: ...
    def CreateEpotRHS(self) -> None: ...
    def CreateEpotResult(self) -> None: ...
    def CreateEpotResultGrad(self) -> None: ...
    def CreatePassiveScalarGradOnBlocks(self) -> None: ...
    def CreatePassiveScalarGradOnBoundaries(self) -> None: ...
    def CreatePassiveScalarOnBlocks(self) -> None: ...
    def CreatePassiveScalarViscosityGrad(self) -> None: ...
    def CreatePressureGradOnBlocks(self) -> None: ...
    def CreatePressureOnBlocks(self) -> None: ...
    def CreatePressureRHS(self) -> None: ...
    def CreatePressureRHSGrad(self) -> None: ...
    def CreatePressureRHSdiv(self) -> None: ...
    def CreatePressureRHSdivGrad(self) -> None: ...
    def CreatePressureResult(self) -> None: ...
    def CreatePressureResultGrad(self) -> None: ...
    def CreateScalarRHS(self) -> None: ...
    def CreateScalarRHSGrad(self) -> None: ...
    def CreateScalarResult(self) -> None: ...
    def CreateScalarResultGrad(self) -> None: ...
    def CreateVelocityGradOnBlocks(self) -> None: ...
    def CreateVelocityGradOnBoundaries(self) -> None: ...
    def CreateVelocityOnBlocks(self) -> None: ...
    def CreateVelocityRHS(self) -> None: ...
    def CreateVelocityRHSGrad(self) -> None: ...
    def CreateVelocityResult(self) -> None: ...
    def CreateVelocityResultGrad(self) -> None: ...
    def CreateVelocitySourceGradOnBlocks(self) -> None:
        """
        Only create grad if velocity source exists, clears it otherwise
        """
    def CreateViscosityGrad(self) -> None: ...
    def Detach(self) -> None: ...
    def DetachFwd(self) -> None: ...
    def DetachGrad(self) -> None: ...
    def GetBoundaryFluxBalance(self) -> torch.Tensor: ...
    def GetCoordinateOrientation(self) -> int:
        """
        Check if all coordinate systems have the same orientation/handedness by checking the sign of the determinant.Returns the sign if all have the same orientation, 0 otherwise.
        """
    def IsInitialized(self) -> bool: ...
    def IsTensorChanged(self) -> bool: ...
    def PrepareSolve(self) -> None: ...
    def SetupEpotOnDomain(
        self, nonOrthoFlags: typing.SupportsInt = 0, useFaceTransform: bool = False
    ) -> None:
        """
        Full epot setup (alloc + matrix fill + block epots). Stores flags so Copy/Clone+PrepareSolve auto-rebuilds.
        """
    def UpdateDomainData(self) -> None: ...
    def __init__(
        self,
        spatialDims: typing.SupportsInt,
        viscosity: torch.Tensor,
        name: str,
        dtype: typing.Any,
        device: torch.device,
        passiveScalarChannels: typing.SupportsInt = 1,
        scalarViscosity: torch.Tensor | None = None,
    ) -> None: ...
    def __str__(self) -> str: ...
    def clearPassiveScalarViscosityGrad(self) -> None: ...
    def clearScalarViscosity(self) -> None: ...
    def clearViscosityGrad(self) -> None: ...
    def getAdvectionScheme(self) -> AdvectionScheme: ...
    def getBlock(self, arg0: typing.SupportsInt) -> Block: ...
    def getBlocks(self) -> list[Block]: ...
    def getDevice(self) -> torch.device: ...
    def getDtype(self) -> typing.Any: ...
    def getMaxVelocity(
        self, withBounds: bool = True, computational: bool = False
    ) -> torch.Tensor: ...
    def getMaxVelocityMagnitude(
        self, withBounds: bool = True, computational: bool = False
    ) -> torch.Tensor: ...
    def getNumBlocks(self) -> int: ...
    def getPassiveScalarChannels(self) -> int: ...
    def getSpatialDims(self) -> int: ...
    def getTotalSize(self) -> int: ...
    def getVertexCoordinates(self) -> list[torch.Tensor]:
        """
        Returns a list of vertex coordnate tensors
        """
    def hasBlockViscosity(self) -> bool:
        """
        Check if ANY block has viscosity set.
        """
    def hasEpot(self) -> bool: ...
    def hasEpotResultGrad(self) -> bool: ...
    def hasPassiveScalar(self) -> bool: ...
    def hasPassiveScalarBlockViscosity(self) -> bool:
        """
        Check if ANY block has passive scalar viscosity set.
        """
    def hasPassiveScalarViscosity(self) -> bool: ...
    def hasPassiveScalarViscosityGrad(self) -> bool: ...
    def hasPrescribedBoundary(self) -> bool: ...
    def hasVertexCoordinates(self) -> bool: ...
    def hasViscosityGrad(self) -> bool: ...
    def isAllFixedBoundariesPassiveScalarTypeStatic(self) -> bool: ...
    def isPassiveScalarViscosityStatic(self) -> bool: ...
    def setA(self, arg0: torch.Tensor) -> None: ...
    def setAGrad(self, arg0: torch.Tensor) -> None: ...
    def setAdvectionScheme(self, scheme: AdvectionScheme) -> None:
        """
        Convective scheme for the momentum and passive-scalar equations. CENTRAL (default) is the original discretization; LINEAR_UPWIND uses implicit first-order upwind plus an explicit second-order upwind correction.
        """
    def setEpotRHS(self, arg0: torch.Tensor) -> None: ...
    def setEpotResult(self, arg0: torch.Tensor) -> None: ...
    def setEpotResultGrad(self, arg0: torch.Tensor) -> None: ...
    def setPassiveScalarViscosityGrad(self, arg0: torch.Tensor) -> None: ...
    def setPressureRHS(self, arg0: torch.Tensor) -> None: ...
    def setPressureRHSGrad(self, arg0: torch.Tensor) -> None: ...
    def setPressureRHSdiv(self, arg0: torch.Tensor) -> None: ...
    def setPressureRHSdivGrad(self, arg0: torch.Tensor) -> None: ...
    def setPressureResult(self, arg0: torch.Tensor) -> None: ...
    def setPressureResultGrad(self, arg0: torch.Tensor) -> None: ...
    def setScalarRHS(self, arg0: torch.Tensor) -> None: ...
    def setScalarRHSGrad(self, arg0: torch.Tensor) -> None: ...
    def setScalarResult(self, arg0: torch.Tensor) -> None: ...
    def setScalarResultGrad(self, arg0: torch.Tensor) -> None: ...
    def setScalarViscosity(self, arg0: torch.Tensor) -> None: ...
    def setVelocityRHS(self, arg0: torch.Tensor) -> None: ...
    def setVelocityRHSGrad(self, arg0: torch.Tensor) -> None: ...
    def setVelocityResult(self, arg0: torch.Tensor) -> None: ...
    def setVelocityResultGrad(self, arg0: torch.Tensor) -> None: ...
    def setViscosity(self, arg0: torch.Tensor) -> None: ...
    def setViscosityGrad(self, arg0: torch.Tensor) -> None: ...
    @property
    def A(self) -> torch.Tensor: ...
    @property
    def AGrad(self) -> torch.Tensor: ...
    @property
    def C(self) -> CSRmatrix: ...
    @property
    def CGrad(self) -> CSRmatrix: ...
    @property
    def Epot(self) -> CSRmatrix | None: ...
    @property
    def P(self) -> CSRmatrix: ...
    @property
    def PGrad(self) -> CSRmatrix: ...
    @property
    def _packedCPU(self) -> torch.Tensor: ...
    @property
    def _packedGPU(self) -> torch.Tensor: ...
    @property
    def epotRHS(self) -> torch.Tensor | None: ...
    @property
    def epotResult(self) -> torch.Tensor | None: ...
    @property
    def epotResultGrad(self) -> torch.Tensor | None: ...
    @property
    def name(self) -> str: ...
    @property
    def passiveScalarViscosity(self) -> torch.Tensor | None: ...
    @property
    def passiveScalarViscosityGrad(self) -> torch.Tensor | None: ...
    @property
    def pressureRHS(self) -> torch.Tensor: ...
    @property
    def pressureRHSGrad(self) -> torch.Tensor: ...
    @property
    def pressureRHSdiv(self) -> torch.Tensor: ...
    @property
    def pressureRHSdivGrad(self) -> torch.Tensor: ...
    @property
    def pressureResult(self) -> torch.Tensor: ...
    @property
    def pressureResultGrad(self) -> torch.Tensor: ...
    @property
    def scalarRHS(self) -> torch.Tensor: ...
    @property
    def scalarRHSGrad(self) -> torch.Tensor: ...
    @property
    def scalarResult(self) -> torch.Tensor: ...
    @property
    def scalarResultGrad(self) -> torch.Tensor: ...
    @property
    def velocityRHS(self) -> torch.Tensor: ...
    @property
    def velocityRHSGrad(self) -> torch.Tensor: ...
    @property
    def velocityResult(self) -> torch.Tensor: ...
    @property
    def velocityResultGrad(self) -> torch.Tensor: ...
    @property
    def viscosity(self) -> torch.Tensor: ...
    @property
    def viscosityGrad(self) -> torch.Tensor | None: ...

class FixedBoundary(Boundary):
    def CreatePassiveScalar(self, createStatic: bool) -> None: ...
    def CreatePassiveScalarGrad(self) -> None: ...
    def CreateVelocity(self, arg0: bool) -> None: ...
    def CreateVelocityGrad(self) -> None: ...
    def Detach(self) -> None: ...
    def DetachFwd(self) -> None: ...
    def DetachGrad(self) -> None: ...
    def GetFluxes(self) -> torch.Tensor:
        """
        Returns a tensor with the boundary-normal fluxes.
        """
    def __str__(self) -> str: ...
    def clearTransform(self) -> None: ...
    def getDevice(self) -> torch.device: ...
    def getEpotCw(self) -> float: ...
    def getPassiveScalarChannels(self) -> int: ...
    def getSizes(self) -> Int4: ...
    def getSpatialDims(self) -> int: ...
    def getStrides(self) -> Int4: ...
    def getVelocity(self, computational: bool) -> torch.Tensor:
        """
        Returns the velocity, broadcasted to varying shape, and optionally transformed to compuatational space.
        """
    def getVelocityVarying(self, size: Int4 | None = None) -> torch.Tensor:
        """
        Broadcast an existing static velocity to varying shape and returns it. The size argument can be used if no size has been set, otherwise it is ignored. Does not change the boundary's velocity.
        """
    def hasEpotCw(self) -> bool: ...
    def hasEpotDirichlet(self) -> bool: ...
    def hasPassiveScalar(self) -> bool: ...
    def hasSize(self) -> bool: ...
    def hasTransform(self) -> bool: ...
    def isEpotInsulating(self) -> bool: ...
    def isPassiveScalarBoundaryTypeStatic(self) -> bool: ...
    def isPassiveScalarStatic(self) -> bool: ...
    def makeVelocityVarying(self, size: Int4 | None = None) -> None:
        """
        Broadcast an existing static velocity to varying shape and sets it on the boundary. The size argument can be used if no size has been set, otherwise it is ignored.
        """
    def setEpotCw(self, cw: typing.SupportsFloat) -> None:
        """
        Set the wall conductance ratio C_w for thin-wall MHD.
        """
    def setEpotDirichlet(self, dirichlet: bool) -> None:
        """
        Set Dirichlet φ=0 BC on this boundary for the electric potential (use at open outflow planes).
        """
    def setEpotInsulating(self, insulating: bool) -> None:
        """
        Set whether this face is insulating (j_n=0) for the inductionless MHD solve. True (the default) is a solid wall; set False on an open in/outflow plane, where current leaves the domain and closes virtually outside. Cannot be inferred from the boundary type: CloseBoundary makes walls and prescribed-velocity in/outflows alike.
        """
    def setPassiveScalar(self, arg0: torch.Tensor) -> None: ...
    def setPassiveScalarGrad(self, arg0: torch.Tensor) -> None: ...
    @typing.overload
    def setPassiveScalarType(self, arg0: BoundaryConditionType) -> None: ...
    @typing.overload
    def setPassiveScalarType(
        self, arg0: collections.abc.Sequence[BoundaryConditionType]
    ) -> None: ...
    def setTransform(self, arg0: torch.Tensor) -> None: ...
    def setVelocity(self, arg0: torch.Tensor) -> None: ...
    def setVelocityGrad(self, arg0: torch.Tensor) -> None: ...
    def setVelocityType(self, arg0: BoundaryConditionType) -> None: ...
    @property
    def _passiveScalarTypes_tensor(self) -> torch.Tensor | None: ...
    @property
    def isVelocityStatic(self) -> bool: ...
    @property
    def passiveScalar(self) -> torch.Tensor | None: ...
    @property
    def passiveScalarGrad(self) -> torch.Tensor | None: ...
    @property
    def passiveScalarTypes(self) -> list[BoundaryConditionType] | None: ...
    @property
    def transform(self) -> torch.Tensor | None: ...
    @property
    def velocity(self) -> torch.Tensor: ...
    @property
    def velocityGrad(self) -> torch.Tensor: ...
    @property
    def velocityType(self) -> BoundaryConditionType: ...

class Int4:
    def __getitem__(self, arg0: typing.SupportsInt) -> int: ...
    def __init__(
        self,
        x: typing.SupportsInt = 0,
        y: typing.SupportsInt = 0,
        z: typing.SupportsInt = 0,
        w: typing.SupportsInt = 0,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    @property
    def w(self) -> int: ...
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...

class LinearSolverResultInfo:
    def __str__(self) -> str: ...
    @property
    def converged(self) -> bool: ...
    @property
    def finalResidual(self) -> float: ...
    @property
    def isFiniteResidual(self) -> bool: ...
    @property
    def usedIterations(self) -> int: ...

class PeriodicBoundary(Boundary):
    def __str__(self) -> str: ...
    @property
    def type(self) -> BoundaryType: ...

class StaticDirichletBoundary(Boundary):
    def __str__(self) -> str: ...
    def getSpatialDims(self) -> int: ...
    @property
    def boundaryScalar(self) -> torch.Tensor: ...
    @property
    def boundaryVelocity(self) -> torch.Tensor: ...
    @property
    def slip(self) -> torch.Tensor: ...
    @property
    def type(self) -> BoundaryType: ...

class VaryingDirichletBoundary(Boundary):
    def CreatePassiveScalarGrad(self) -> None: ...
    def CreateVelocityGrad(self) -> None: ...
    def __str__(self) -> str: ...
    def clearTransform(self) -> None: ...
    def getSizes(self) -> Int4: ...
    def getSpatialDims(self) -> int: ...
    def getStrides(self) -> Int4: ...
    def setPassiveScalar(self, arg0: torch.Tensor) -> None: ...
    def setPassiveScalarGrad(self, arg0: torch.Tensor) -> None: ...
    def setTransform(self, arg0: torch.Tensor) -> None: ...
    def setVelocity(self, arg0: torch.Tensor) -> None: ...
    def setVelocityGrad(self, arg0: torch.Tensor) -> None: ...
    @property
    def boundaryScalar(self) -> torch.Tensor: ...
    @property
    def boundaryScalarGrad(self) -> torch.Tensor: ...
    @property
    def boundaryVelocity(self) -> torch.Tensor: ...
    @property
    def boundaryVelocityGrad(self) -> torch.Tensor: ...
    @property
    def hasTransform(self) -> bool: ...
    @property
    def slip(self) -> torch.Tensor: ...
    @property
    def transform(self) -> torch.Tensor: ...
    @property
    def type(self) -> BoundaryType: ...

def ComputeCurrentDensityFaceBased(
    domain: Domain, epotField: torch.Tensor, uCrossBField: torch.Tensor
) -> torch.Tensor:
    """
    Compute face-based current density J = -grad(phi) + u×B. Guarantees discrete div(J)=0. Returns J [totalSize*3] (always 3 components).
    """

def ComputeCurrentDensityFaceBasedGrad(
    domain: Domain,
    epotField: torch.Tensor,
    uCrossBField: torch.Tensor,
    gradJ: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backpropagate current density grad to epot and u_cross_eb grads (CUDA)
    """

def ComputeEpotRHS(domain: Domain, vectorField: torch.Tensor) -> torch.Tensor:
    """
    Compute the electric-potential Poisson RHS div(u×B) [totalSize] from the flat u×B field [totalSize*dims]. The normal flux through prescribed boundaries (FIXED/DIRICHLET/DIRICHLET_VARYING/GRADIENT) is dropped for an insulating (j_n=0) outflow.
    """

def ComputeEpotRHSGrad(domain: Domain, gradDivergence: torch.Tensor) -> torch.Tensor:
    """
    Backpropagate scalar divergence grad to vector field grad (CUDA)
    """

def ComputeFieldGradient(domain: Domain, scalarField: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of an arbitrary flat scalar field [totalSize] with Neumann BCs. Returns vector field [totalSize*dims].
    """

def ComputeFieldGradientFVM(domain: Domain, scalarField: torch.Tensor) -> torch.Tensor:
    """
    Compute FVM (Green-Gauss) gradient of a flat scalar field [totalSize]. Uses face-based interpolation consistent with the Laplacian stencil. Returns vector field [totalSize*dims].
    """

def ComputePressureGradient(
    arg0: Domain, arg1: bool, arg2: typing.SupportsInt
) -> torch.Tensor:
    """
    Pressure gradient (CUDA)
    """

def ComputeSpatialVelocityGradients(domain: Domain) -> list[list[torch.Tensor]]:
    """
    Compute the spatial gradients of all velocity components of all blocks in the domain. Returns nested lists of tensors: [Blocks: [Components: NCDHW]]. The outer index selects the velocity component k, the tensor's channel axis C the spatial direction i, i.e. result[block][k][:,i] = d(u_k)/d(x_i).
    """

def ComputeVelocityDivergence(arg0: Domain) -> torch.Tensor:
    """
    Velocity divergence (CUDA)
    """

def CoordsToFaceTransforms(arg0: torch.Tensor) -> torch.Tensor:
    """
    Computes cell face transformation metrics from cell vertex coordinates.
    """

def CoordsToTransforms(arg0: torch.Tensor) -> torch.Tensor:
    """
    Computes cell transformation metrics from cell vertex coordinates.
    """

def CopyEpotResultFromBlocks(arg0: Domain) -> None:
    """
    Copy block.epot to domain.epotResult (CUDA)
    """

def CopyEpotResultGradFromBlocks(arg0: Domain) -> None:
    """
    Copy block.epot_grad to domain.epotResult_grad (CUDA)
    """

def CopyEpotResultToBlocks(arg0: Domain) -> None:
    """
    Copy domain.epotResult to block.epot (CUDA)
    """

def CopyPressureResultFromBlocks(arg0: Domain) -> None:
    """
    PISO copy pressure result (CUDA)
    """

def CopyPressureResultGradFromBlocks(arg0: Domain) -> None:
    """
    PISO copy scalar result grad (CUDA)
    """

def CopyPressureResultToBlocks(arg0: Domain) -> None:
    """
    PISO copy pressure result (CUDA)
    """

def CopyScalarResultFromBlocks(arg0: Domain) -> None:
    """
    PISO copy scalar result (CUDA)
    """

def CopyScalarResultGradFromBlocks(arg0: Domain) -> None:
    """
    PISO copy scalar result grad (CUDA)
    """

def CopyScalarResultGradToBlocks(arg0: Domain) -> None:
    """
    PISO copy scalar result grad (CUDA)
    """

def CopyScalarResultToBlocks(arg0: Domain) -> None:
    """
    PISO copy scalar result (CUDA)
    """

def CopyVelocityResultFromBlocks(arg0: Domain) -> None:
    """
    PISO copy velocity result back (CUDA)
    """

def CopyVelocityResultGradFromBlocks(arg0: Domain) -> None:
    """
    PISO copy velocity result grad (CUDA)
    """

def CopyVelocityResultGradToBlocks(arg0: Domain) -> None:
    """
    PISO copy velocity result grad (CUDA)
    """

def CopyVelocityResultToBlocks(arg0: Domain) -> None:
    """
    PISO copy velocity result (CUDA)
    """

def CorrectVelocity(
    domain: Domain,
    timeStep: torch.Tensor,
    version: typing.SupportsInt = 0,
    timeStepNorm: bool = False,
) -> None:
    """
    Correct the velocity given by pressureRHSdiv with the gradient of the blocks' pressure fields.
    """

def CorrectVelocityGrad(arg0: Domain, arg1: torch.Tensor, arg2: bool) -> None:
    """
    PISO correct velocity gradient (CUDA)
    """

def EigenDecomposition(
    matrices: torch.Tensor,
    outputEigenvalues: bool = True,
    outputEigenvectors: bool = True,
    normalizeEigenvectors: bool = False,
) -> list[torch.Tensor | None]:
    """
    Eigen decomposition of symmetric matrices (CUDA). Returns tensors eigenvalues and eigenvectors.
    """

def InvertMatrix(matrices: torch.Tensor, inPlace: bool = False) -> torch.Tensor:
    """
    Returns the inverse of a square matrix in the channel dimension.
    """

def MakeBasisUnique(
    basisMatrices: torch.Tensor, sortingVectors: torch.Tensor, inPlace: bool = False
) -> torch.Tensor:
    """
    Makes a given set of orthogonal basis vectors (as colums in flat row-major matrices) unique (CUDA).
    """

def MakeCoordsNDNonUniformScaleNormalized(
    arg0: typing.SupportsInt,
    arg1: typing.SupportsInt,
    arg2: typing.SupportsInt,
    arg3: torch.Tensor,
) -> torch.Tensor:
    """
    MakeCoordsNDNonUniformScaleNormalized (CUDA)
    """

def MakeGrid2DNonUniformScale(
    arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: torch.Tensor
) -> torch.Tensor:
    """
    MakeGrid2DNonUniformScale (CUDA)
    """

def MakeGridNDExpScaleNormalized(
    arg0: typing.SupportsInt,
    arg1: typing.SupportsInt,
    arg2: typing.SupportsInt,
    arg3: torch.Tensor,
) -> torch.Tensor:
    """
    MakeGridNDExpScaleNormalized (CUDA)
    """

def MakeGridNDNonUniformScaleNormalized(
    arg0: typing.SupportsInt,
    arg1: typing.SupportsInt,
    arg2: typing.SupportsInt,
    arg3: torch.Tensor,
) -> torch.Tensor:
    """
    MakeGridNDNonUniformScaleNormalized (CUDA)
    """

def SGSviscosityIncompressibleSmagorinsky(
    domain: Domain, coefficient: torch.Tensor
) -> list[torch.Tensor]:
    """
    Compute the additive viscosities for a Smagorinsky SGS scheme based on the velocity field. Returns a list of viscosity tensors, one per block. NOTE: 'coefficient' is already C_s^2 (not C_s), and the filter width used is the max cell edge length rather than cellVolume^(1/dims).
    """

def SGSviscosityIncompressibleWALE(
    domain: Domain, coefficient: torch.Tensor
) -> list[torch.Tensor]:
    """
    Compute the additive viscosities for a WALE (Wall-Adapting Local Eddy-viscosity, Nicoud & Ducros 1999) SGS scheme based on the velocity field. Returns a list of viscosity tensors, one per block. 'coefficient' is the textbook Cw (typ. 0.325-0.5) and is SQUARED internally: nu_t = (Cw*Delta)^2 * (Sd:Sd)^1.5 / ((S:S)^2.5 + (Sd:Sd)^1.25), with Delta = cellVolume^(1/dims). NOTE: unlike SGSviscosityIncompressibleSmagorinsky, whose 'coefficient' is already C_s^2, this takes Cw, not Cw^2. Only meaningful in 3D: the WALE operator vanishes identically for divergence-free 1D/2D fields.
    """

def SampleTransformedGridGlobalToLocal(
    globalData: torch.Tensor,
    globalTransform: torch.Tensor,
    localCoords: torch.Tensor,
    boundarySamplingMode: BoundarySampling,
    constantValue: torch.Tensor,
) -> torch.Tensor:
    """
    Sample from a globally transformed grid (single transformation matrix for all cells) to a locally transformed grid (individual cell coordinates).Each cell of the output grid gathers its value by interpolating cells of the input grid. This can lead to aliasing.
    """

def SampleTransformedGridLocalToGlobal(
    localData: torch.Tensor,
    localCoords: torch.Tensor,
    globalTransform: torch.Tensor,
    globalShape: torch.Tensor,
    fillMaxSteps: typing.SupportsInt = 0,
) -> list[torch.Tensor]:
    """
    Sample from a locally transformed grid (individual cell coordinates) to a globally transformed grid (single transformation matrix for all cells).Each cell of the input grid scatters its value to cells of the output grid. The output grid is then normalized using the accumulated scattering weights.This can lead to empty (zero) cells in the output.Returns the output grid and scattering weights.
    """

def SampleTransformedGridLocalToGlobalMulti(
    localData: collections.abc.Sequence[torch.Tensor],
    localCoords: collections.abc.Sequence[torch.Tensor],
    globalTransform: torch.Tensor,
    globalShape: torch.Tensor,
    fillMaxSteps: typing.SupportsInt = 0,
) -> list[torch.Tensor]:
    """
    Version of SampleTransformedGridLocalToGlobal with multiple input grids.
    """

def SetupAdvectionMatrix(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoFlags: typing.SupportsInt,
    forPassiveScalar: bool = False,
    passiveScalarChannel: typing.SupportsInt = 0,
) -> None:
    """
    Setup the matrix of the advection/diffusion system. Used for both velocity and passive scalars.
    """

def SetupAdvectionMatrixGrad(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoFlags: typing.SupportsInt,
    forPassiveScalar: bool = False,
    passiveScalarChannel: typing.SupportsInt = 0,
) -> None:
    """
    PISO setup A matrix gradient (CUDA)
    """

def SetupAdvectionScalar(
    domain: Domain, timeStep: torch.Tensor, nonOrthoFlags: typing.SupportsInt
) -> None:
    """
    Setup the RHS for passive scalar advection/diffusion.
    """

def SetupAdvectionScalarGrad(
    domain: Domain, timeStep: torch.Tensor, nonOrthoFlags: typing.SupportsInt
) -> None:
    """
    PISO setup scalar advection gradient (CUDA)
    """

def SetupAdvectionVelocity(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoFlags: typing.SupportsInt,
    applyPressureGradient: bool = False,
) -> None:
    """
    Setup the RHS for velocity advection/diffusion.
    """

def SetupAdvectionVelocityGrad(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoFlags: typing.SupportsInt,
    applyPressureGradient: bool = False,
) -> None:
    """
    PISO setup velocity advection gradient (CUDA)
    """

def SetupEpotMatrix(
    domain: Domain, nonOrthoMode: typing.SupportsInt = 0, useFaceTransform: bool = False
) -> None:
    """
    Build the Laplacian matrix for the electric potential Poisson equation (MHD). For thin-wall FIXED boundaries (epotCw>0) the Robin BC dphi/dn=Cw*nabla2_tau(phi) is incorporated directly — no augmented system required. Requires domain.CreateEpotMatrix() to have been called.
    """

def SetupPressureCorrection(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """
    Setup matrix, RHS, and div(RHS) of the pressure system.
    """

def SetupPressureCorrectionGrad(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """
    PISO setup pressure gradient (CUDA)
    """

def SetupPressureMatrix(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
) -> None:
    """
    Setup only the matrix of the pressure system.
    """

def SetupPressureMatrixGrad(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
) -> None:
    """
    PISO setup pressure gradient (CUDA)
    """

def SetupPressureRHS(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """
    Setup RHS and div(RHS) of the pressure system.
    """

def SetupPressureRHSGrad(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """
    PISO setup pressure gradient (CUDA)
    """

def SetupPressureRHSdiv(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """
    Compute only div(RHS).
    """

def SetupPressureRHSdivGrad(
    domain: Domain,
    timeStep: torch.Tensor,
    nonOrthoMode: typing.SupportsInt = 0,
    useFaceTransform: bool = False,
    timeStepNorm: bool = False,
) -> None:
    """
    PISO setup pressure gradient (CUDA)
    """

def SolveLinear(
    A: CSRmatrix,
    RHS: torch.Tensor,
    x: torch.Tensor,
    maxIterations: torch.Tensor = 1000,
    tolerance: torch.Tensor = 1e-08,
    convergenceCriterion: ConvergenceCriterion = ConvergenceCriterion.NORM2_NORMALIZED,
    useBiCG: bool = False,
    matrixRankDeficient: bool = False,
    residualResetSteps: typing.SupportsInt = 0,
    transposeA: bool = False,
    printResidual: bool = False,
    returnBestResult: bool = False,
    BiCGwithPreconditioner: bool = True,
) -> list[LinearSolverResultInfo]:
    """
    Sparse linear solve on GPU (CUDA). With option for CG, BiCGStab, and preconditioning.
    """

def SparseOuterProduct(arg0: torch.Tensor, arg1: torch.Tensor, arg2: CSRmatrix) -> None:
    """
    Outer product multiplied by sparsity pattern of result matrix. (CUDA)
    """

def TransformVectors(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: bool
) -> torch.Tensor:
    """
    Transform vectors with given transformation. (CUDA)
    """

def VectorToDiagMatrix(vectors: torch.Tensor) -> torch.Tensor:
    """
    Writes the vector in the channel dimension into the diagonal of a flat row-major matrix in the channel dimension.
    """

def matmul(
    vectorMatrixA: torch.Tensor,
    vectorMatrixB: torch.Tensor,
    transposeA: bool = False,
    invertA: bool = False,
    transposeB: bool = False,
    invertB: bool = False,
    transposeOutput: bool = False,
    invertOutput: bool = False,
) -> torch.Tensor:
    """
    Computes the product matrix/vector * matrix/vector for vectors or symmetrics matrices given in the channel dimension of NCDHW tensors.Matrices are assumed to be in a flattend row-major format.transpose and invert only apply if the quantity is a matrix.
    """

def matmulGrad(
    vectorMatrixA: torch.Tensor,
    vectorMatrixB: torch.Tensor,
    outputGrad: torch.Tensor,
    transposeA: bool = False,
    invertA: bool = False,
    transposeB: bool = False,
    invertB: bool = False,
    transposeOutput: bool = False,
    invertOutput: bool = False,
) -> list[torch.Tensor]:
    """
    Computes gradients of the product matrix/vector * matrix/vector for vectors or symmetrics matrices given in the channel dimension of NCDHW tensors.Matrices are assumed to be in a flattend row-major format.transpose and invert only apply if the quantity is a matrix. Gradients for inverted matrices are not computed.
    """

ABS_MAX: ConvergenceCriterion  # value = <ConvergenceCriterion.ABS_MAX: 4>
ABS_MEAN: ConvergenceCriterion  # value = <ConvergenceCriterion.ABS_MEAN: 3>
ABS_SUM: ConvergenceCriterion  # value = <ConvergenceCriterion.ABS_SUM: 2>
CENTRAL: AdvectionScheme  # value = <AdvectionScheme.CENTRAL: 0>
CLAMP: BoundarySampling  # value = <BoundarySampling.CLAMP: 1>
CONNECTED: BoundaryType  # value = <BoundaryType.CONNECTED: 20>
CONSTANT: BoundarySampling  # value = <BoundarySampling.CONSTANT: 0>
DIRICHLET: BoundaryConditionType  # value = <BoundaryConditionType.DIRICHLET: 0>
DIRICHLET_VARYING: BoundaryType  # value = <BoundaryType.DIRICHLET_VARYING: 1>
FIXED: BoundaryType  # value = <BoundaryType.FIXED: 30>
LINEAR_UPWIND: AdvectionScheme  # value = <AdvectionScheme.LINEAR_UPWIND: 1>
NEUMANN: BoundaryConditionType  # value = <BoundaryConditionType.NEUMANN: 1>
NORM2: ConvergenceCriterion  # value = <ConvergenceCriterion.NORM2: 0>
NORM2_NORMALIZED: (
    ConvergenceCriterion  # value = <ConvergenceCriterion.NORM2_NORMALIZED: 1>
)
PERIODIC: BoundaryType  # value = <BoundaryType.PERIODIC: 21>
