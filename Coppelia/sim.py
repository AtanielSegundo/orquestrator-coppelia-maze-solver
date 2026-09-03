from typing import Optional
from Coppelia.base import ObjectHandle,CoppeliaConsts

class CoppeliaSimulation(CoppeliaConsts):
    def getObject(self,objectPath:str,options:dict=None) -> Optional[ObjectHandle]:
        """Retrieves an object handle based on its path and alias

        Args:
            objectPath (str): path and alias of the object to find. See the section on accessing scene objects for details.
            options (dict, optional): optional map indicating how the object should be searched:
                noError(bool): 
                    can generate a silent error if an object was not found
                index(int): 
                    fetches the n-th object (zero based index) that satisfies the specified objectPath
                proxy(int):
                    allows to search the hierarchy tree of the specified proxy object

        Returns:
            Optional[ObjectHandle]: handle of object or None
        """
    
    def setObjectInt32Param(self,
                        objectHandle: ObjectHandle,
                        parameterID: int,
                        parameter: int) -> None:
        """Sets an int32 parameter of a scene object.

        Deprecated em versões recentes do CoppeliaSim — veja a API de
        'properties'. Ainda funcional, mas pode ser removida no futuro.

        Args:
            objectHandle (ObjectHandle): handle do objeto alvo.
            parameterID (int): identificador do parâmetro a definir. Use as
                constantes sim.* correspondentes, p.ex.:
                    sim.shapeintparam_static         -> shape estática (0/1)
                    sim.shapeintparam_respondable    -> responde a colisões (0/1)
                    sim.objintparam_visibility_layer -> camada de visibilidade
            parameter (int): o valor inteiro a atribuir (frequentemente 0/1
                para parâmetros booleanos).

        Returns:
            None
        """
