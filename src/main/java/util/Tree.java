package util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by ps324 on 30/07/2014.
 */
public class Tree<T> {
    private Node<T> root;

    public Tree(T rootData) {
        root = new Node<T>();
        root.data = rootData;
        root.children = new ArrayList<Node<T>>();
    }

    public Tree(Node<T> node){
        root = node;
        root.children = new ArrayList<Node<T>>();
    }

    public static class Node<T> {
        private T data;
        private int[] pointers;
        private Node<T> parent;
        private List<Node<T>> children;

        public Node() {
            this.children = new ArrayList<Node<T>>();
        }

        public Node(T nodeData){
            this.data = nodeData;
            this.children = new ArrayList<Node<T>>();
        }

        public List<Node<T>> getChildren() {
            return children;
        }

        public void addChild(double[] child) {
            Tree.Node n = new Tree.Node(child);
            n.setParent(this);
            this.children.add(n);
        }

        public void addChild(Node<T> child) {
            child.setParent(this);
            this.children.add(child);
        }

        public void setChildren(List<Node<T>> children) {
            this.children = children;
        }

        public Node<T> getParent() {
            return parent;
        }

        public void setParent(Node<T> parent) {
            this.parent = parent;
        }

        public T getData() {
            return data;
        }

        public void setData(T data, int[] backPointers) {
            this.data = data;
            this.pointers = backPointers;
        }

        public int[] getPointers() {
            return pointers;
        }

        public void setPointers(int[] pointers) {
            this.pointers = pointers;
        }

        public String toString(){
            return Arrays.toString((double[])this.data);
        }
    }

    public Node<T> getRoot() {
        return root;
    }

    public void setRoot(Node<T> root) {
        this.root = root;
    }


}