class Taxonomy:
    """
    This class is responsible for creating a taxonomy.json from labels.
    It provides methods to create a node, find a node, convert labels into a taxonomy, and get the taxonomy.
    """

    def __init__(self, labels):
        self.labels = labels
        self.nodes = []

        self.convert()

    def _create_node(self, nodeName: str, parentName: str = None, parentId: str = None):
        return {
            'parentName': parentName,
            'nodeName': nodeName,
            'nodeId': len(self.nodes),
            'parentId': parentId
        }

    def _find_node(self, node_name: str):
        for node in self.nodes:
            if node['nodeName'] == node_name:
                return (node['nodeId'], node['nodeName'])
        return None

    def convert(self):
        self.nodes.append(self._create_node('Root'))

        parent_id = 1

        if self.labels:  # Ensure self.labels is not empty
            print("First item in self.labels:", self.labels[0])

        second_node_names = {label.label for label in self.labels}

        for second_node_name in second_node_names:
            self.nodes.append(self._create_node(
                second_node_name, 'Root', parent_id))

        for label in self.labels:
            parent_id, parent_name = self._find_node(label.label)
            for annotation in label.annotations:
                self.nodes.append(self._create_node(
                    annotation['label'], parent_id, parent_name)
                )

    def get(self):
        return self.nodes
